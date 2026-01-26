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

#include "ucxx_sender_op.hpp"

#include <holoscan/holoscan.hpp>

#include <operators/ucxx_send_receive/serialize_tensor.hpp>

namespace holoscan::ops {

void UcxxSenderOp::setup(holoscan::OperatorSpec& spec) {
  spec.param(tag_, "tag", "Tag", "UCX tag number", 0ul);
  spec.param(endpoint_, "endpoint", "Endpoint", "UcxxEndpoint resource");
  spec.param(allocator_, "allocator", "Allocator", "Allocator for staging buffers");
  spec.param(blocking_,
             "blocking",
             "Blocking",
             "If true, do not execute until endpoint is connected. If false (default), drain "
             "inputs and drop sends while disconnected (prevents upstream backpressure).",
             false);
  spec.input<holoscan::gxf::Entity>("in");

  bool blocking_value = false;
  for (auto arg : args()) {
    if (arg.name() == "blocking") {
      blocking_value = std::any_cast<bool>(arg.value());
      break;
    }
  }

  // If blocking == true: add the endpoint's is_alive_condition to this operator so that it will
  // only execute when the endpoint is alive.
  if (blocking_value) {
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
}

void UcxxSenderOp::compute(holoscan::InputContext& input, holoscan::OutputContext&,
                           holoscan::ExecutionContext&) {
  auto in_message = input.receive<holoscan::gxf::Entity>("in").value();

  // Always clean up completed requests (even when disconnected).
  for (auto it = requests_.begin(); it != requests_.end();) {
    // Check if both header and data requests are completed
    bool header_done = !it->header_request || it->header_request->isCompleted();
    bool data_done = !it->data_request || it->data_request->isCompleted();
    if (!header_done || !data_done) {
      ++it;
      continue;
    }

    // Check status of both requests
    auto check_status = [](const std::shared_ptr<::ucxx::Request>& req, const char* name) {
      if (!req) return;
      if (ucs_status_t status = req->getStatus(); status != UCS_OK) {
        if (status == UCS_ERR_CONNECTION_RESET || status == UCS_ERR_NOT_CONNECTED ||
            status == UCS_ERR_UNREACHABLE || status == UCS_ERR_CANCELED) {
          HOLOSCAN_LOG_WARN("{} send failed (likely disconnect/reconnect) with status: {}",
                            name, ucs_status_string(status));
        } else {
          HOLOSCAN_LOG_ERROR("{} send failed with status: {}", name, ucs_status_string(status));
        }
      }
    };
    check_status(it->header_request, "Header");
    check_status(it->data_request, "Data");
    it = requests_.erase(it);
  }

  // Snapshot the UCXX endpoint for the duration of this tick.
  //
  // This avoids a race where the endpoint is reset concurrently (e.g., when the subscriber closes)
  // after the connectivity check but before tagSend() is invoked, which can otherwise lead to
  // calling tagSend() with a null endpoint.
  auto endpoint_resource = endpoint_.get();
  std::shared_ptr<::ucxx::Endpoint> ucxx_endpoint =
      endpoint_resource ? endpoint_resource->endpoint() : nullptr;

  // If not connected (waiting for subscriber to connect, or after disconnect),
  // drop the message to avoid stalling upstream operators like Holoviz.
  if (!ucxx_endpoint) {
    // Ensure we don't keep stale in-flight requests around while disconnected.
    requests_.clear();
    return;
  }

  // Try to get tensor - first as holoscan::Tensor, then as nvidia::gxf::Tensor
  // Use a pointer to handle both cases uniformly
  nvidia::gxf::Tensor* gxf_tensor_ptr = nullptr;
  std::shared_ptr<nvidia::gxf::Tensor> gxf_tensor_storage;  // For holoscan::Tensor case

  const char* tensor_name = "";

  auto maybe_holoscan_tensor = in_message.get<holoscan::Tensor>(tensor_name);
  if (maybe_holoscan_tensor) {
    // Convert holoscan::Tensor to nvidia::gxf::Tensor for serialization
    gxf_tensor_storage = std::make_shared<nvidia::gxf::Tensor>(maybe_holoscan_tensor->dl_ctx());
    if (!gxf_tensor_storage) {
      HOLOSCAN_LOG_ERROR("Failed to convert holoscan::Tensor to nvidia::gxf::Tensor");
      return;
    }
    gxf_tensor_ptr = gxf_tensor_storage.get();
  } else {
    // Try to get nvidia::gxf::Tensor directly by casting to nvidia::gxf::Entity
    auto maybe_gxf_tensor =
        static_cast<nvidia::gxf::Entity&>(in_message).get<nvidia::gxf::Tensor>(tensor_name);
    if (!maybe_gxf_tensor) {
      HOLOSCAN_LOG_ERROR("Failed to get tensor from input message (tried both "
                         "holoscan::Tensor and nvidia::gxf::Tensor)");
      return;
    }
    // Use the GXF tensor directly (Handle provides pointer-like access)
    gxf_tensor_ptr = maybe_gxf_tensor.value().get();
  }

  // Create a send request for two-phase transfer
  SendRequest& send = requests_.emplace_back();

  // Build header from tensor metadata (no data copy)
  send.header = holoscan::ops::ucxx::buildTensorHeader(*gxf_tensor_ptr);

  // Phase 1: Send header (CPU) - runs in parallel with phase 2
  send.header_request = ucxx_endpoint->tagSend(
      &send.header, sizeof(holoscan::ops::ucxx::TensorHeader), ::ucxx::Tag{tag_.get()});

  // Phase 2: Send tensor data (GPU or CPU pointer, UCX handles both)
  const size_t tensor_size =
      gxf_tensor_ptr->element_count() * gxf_tensor_ptr->bytes_per_element();
  send.data_request = ucxx_endpoint->tagSend(
      gxf_tensor_ptr->pointer(), tensor_size, ::ucxx::Tag{tag_.get()});
}

}  // namespace holoscan::ops
