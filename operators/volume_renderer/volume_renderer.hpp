/* SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef VOLUME_RENDERER_VOLUME_RENDERER
#define VOLUME_RENDERER_VOLUME_RENDERER

#include "holoscan/holoscan.hpp"

namespace holoscan::ops {

class VolumeRendererOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(VolumeRendererOp)

  void initialize() override;
  void start() override;
  void stop() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace holoscan::ops

#endif /* VOLUME_RENDERER_VOLUME_RENDERER */
