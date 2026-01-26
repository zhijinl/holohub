/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ucxx_receiver_op.hpp"

#include <cstring>

#include <cuda_runtime.h>
#include <fmt/format.h>
#include <holoscan/holoscan.hpp>

#include <operators/ucxx_send_receive/serialize_tensor.hpp>

namespace holoscan::ops {

void UcxxReceiverOp::setup(holoscan::OperatorSpec& spec) {
  spec.param(tag_, "tag", "Tag", "UCX tag number", 0ul);
  spec.param(buffer_size_, "buffer_size", "Buffer size",
             "Tensor data buffer size in bytes");  // Required, no default
  spec.param(receive_on_device_, "receive_on_device", "Receive on device",
             "Allocate tensor buffer on device (GPU) if true, host (CPU) if false", true);
  spec.param(endpoint_, "endpoint", "Endpoint", "UcxxEndpoint resource");
  spec.param(allocator_, "allocator", "Allocator", "Allocator for tensor buffers");
  spec.output<holoscan::gxf::Entity>("out");

  // Create tensor received condition for async tensor data receive
  tensor_received_condition_ = fragment()->make_condition<holoscan::AsynchronousCondition>(
      fmt::format("{}_tensor_received", name()));
  add_arg(tensor_received_condition_);

  // Add the endpoint's is_alive_condition to this operator so that it will execute only when
  // the endpoint is alive.
  for (auto arg : args()) {
    if (arg.name() == "endpoint") {
      auto resource = std::any_cast<std::shared_ptr<holoscan::Resource>>(arg.value());
      auto endpoint = std::dynamic_pointer_cast<UcxxEndpoint>(resource);
      if (endpoint) {
        add_arg(endpoint->is_alive_condition());
      } else {
        HOLOSCAN_LOG_ERROR("Failed to cast endpoint resource to UcxxEndpoint");
      }
      break;
    }
  }
}

void UcxxReceiverOp::start() {
  // Ensure CUDA context is active on this thread so UCX can detect GPU memory.
  // The CUDA primary context is shared across threads, but must be explicitly
  // activated on each thread for cuPointerGetAttribute() to work.
  cudaFree(0);
}

void UcxxReceiverOp::stop() {
  if (header_request_) {
    header_request_->cancel();
  }
  if (tensor_request_) {
    tensor_request_->cancel();
  }
}

void UcxxReceiverOp::compute([[maybe_unused]] holoscan::InputContext& input,
                             holoscan::OutputContext& output,
                             holoscan::ExecutionContext& context) {
  // Wait for both header and tensor data to complete
  if (header_request_ && header_request_->isCompleted() &&
      tensor_request_ && tensor_request_->isCompleted()) {
    auto header_status = header_request_->getStatus();
    auto data_status = tensor_request_->getStatus();

    if (header_status == UCS_OK && data_status == UCS_OK) {
      // Parse header
      const holoscan::ops::ucxx::TensorHeader* header =
          reinterpret_cast<const holoscan::ops::ucxx::TensorHeader*>(header_buffer_.data());

      // Create output tensor using received buffer
      auto out_entity = holoscan::gxf::Entity::New(&context);
      auto tensor_handle =
          static_cast<nvidia::gxf::Entity&>(out_entity).add<nvidia::gxf::Tensor>("");
      if (!tensor_handle) {
        HOLOSCAN_LOG_ERROR("Failed to add tensor to entity");
        tensor_buffer_ = nullptr;
        tensor_request_ = nullptr;
        header_request_ = nullptr;
        return;
      }

      // Build shape and strides from header
      std::vector<int32_t> dims(header->dims, header->dims + header->rank);
      nvidia::gxf::Shape shape(dims);
      nvidia::gxf::Tensor::stride_array_t strides;
      std::memcpy(strides.data(), header->strides, sizeof(header->strides));

      // Wrap the received buffer in the tensor
      auto buffer_ref = tensor_buffer_;
      auto result = tensor_handle.value()->wrapMemory(
          shape,
          header->element_type,
          header->bytes_per_element,
          strides,
          receive_on_device_.get() ? nvidia::gxf::MemoryStorageType::kDevice
                                   : nvidia::gxf::MemoryStorageType::kHost,
          tensor_buffer_.get(),
          [buffer_ref](void*) mutable {
            buffer_ref.reset();
            return nvidia::gxf::Success;
          });

      if (!result) {
        HOLOSCAN_LOG_ERROR("Failed to wrap memory in tensor");
        tensor_buffer_ = nullptr;
        tensor_request_ = nullptr;
        header_request_ = nullptr;
        return;
      }

      output.emit(out_entity, "out");
    } else {
      if (header_status != UCS_OK) {
        HOLOSCAN_LOG_ERROR("Header receive failed with status: {}",
                           ucs_status_string(header_status));
      }
      if (data_status != UCS_OK) {
        HOLOSCAN_LOG_ERROR("Data receive failed with status: {}",
                           ucs_status_string(data_status));
      }
    }

    tensor_buffer_ = nullptr;
    tensor_request_ = nullptr;
    header_request_ = nullptr;
  }

  // Post new receive requests (both header and tensor in parallel)
  if (!header_request_) {
    // Snapshot the UCXX endpoint for the duration of this tick to avoid races with disconnects.
    auto endpoint_resource = endpoint_.get();
    std::shared_ptr<::ucxx::Endpoint> ucxx_endpoint =
        endpoint_resource ? endpoint_resource->endpoint() : nullptr;
    if (!ucxx_endpoint) { return; }

    // Post header receive
    async_condition()->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);
    header_request_ = ucxx_endpoint->tagRecv(
        header_buffer_.data(), header_buffer_.size(), ::ucxx::Tag{tag_.get()}, ::ucxx::TagMaskFull,
        /*enablePythonFuture=*/false, [this](ucs_status_t, std::shared_ptr<void>) {
          async_condition()->event_state(holoscan::AsynchronousEventState::EVENT_DONE);
        });

    // Allocate buffer for tensor data (GPU or Host based on receive_on_device_)
    tensor_buffer_ = std::shared_ptr<nvidia::byte>(
        static_cast<nvidia::byte*>(
            allocator_.get()->allocate(buffer_size_.get(),
                                       receive_on_device_.get()
                                           ? holoscan::MemoryStorageType::kDevice
                                           : holoscan::MemoryStorageType::kHost)),
        [this](nvidia::byte* ptr) { allocator_.get()->free(ptr); });

    // Post tensor data receive (GPU buffer)
    tensor_received_condition_->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);
    tensor_request_ = ucxx_endpoint->tagRecv(
        tensor_buffer_.get(), buffer_size_.get(),
        ::ucxx::Tag{tag_.get()}, ::ucxx::TagMaskFull,
        /*enablePythonFuture=*/false, [this](ucs_status_t, std::shared_ptr<void>) {
          tensor_received_condition_->event_state(holoscan::AsynchronousEventState::EVENT_DONE);
        });
  }
}

}  // namespace holoscan::ops
