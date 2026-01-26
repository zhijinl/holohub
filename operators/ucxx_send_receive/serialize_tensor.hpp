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

#pragma once

#include <cstdint>
#include <cstddef>

#include "holoscan/holoscan.hpp"
#include "gxf/std/tensor.hpp"

namespace holoscan::ops::ucxx {

using nvidia::gxf::MemoryStorageType;
using nvidia::gxf::PrimitiveType;
using nvidia::gxf::Shape;

#pragma pack(push, 1)
struct TensorHeader {
  MemoryStorageType storage_type;     // CPU or GPU tensor
  PrimitiveType element_type;         // Tensor element type
  uint64_t bytes_per_element;         // Bytes per tensor element
  uint32_t rank;                      // Tensor rank
  int32_t dims[Shape::kMaxRank];      // Tensor dimensions
  uint64_t strides[Shape::kMaxRank];  // Tensor strides
};
#pragma pack(pop)

// Build TensorHeader from tensor metadata (no data copy).
//
// This function creates a TensorHeader containing the tensor's shape, stride,
// and type information without copying the actual tensor data. This is useful
// for zero-copy transfers where the header is sent separately from the data.
//
// Args:
//   tensor: The tensor to build the header from.
//
// Returns:
//   TensorHeader containing the tensor's metadata.
TensorHeader buildTensorHeader(const nvidia::gxf::Tensor& tensor);

}  // namespace holoscan::ops::ucxx
