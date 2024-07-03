/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dpdk_bench_op_rx.h"
#include "dpdk_bench_op_tx.h"
#include "doca_bench_op_rx.h"
#include "doca_bench_op_tx.h"
#include "adv_network_kernels.h"
#include "holoscan/holoscan.hpp"
#include <assert.h>
#include <sys/time.h>

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    HOLOSCAN_LOG_INFO("Initializing advanced network operator");
    const auto [rx_en, tx_en] = holoscan::ops::adv_net_get_rx_tx_cfg_en(config());
    const std::string mgr = holoscan::ops::adv_net_get_manager(config());
    HOLOSCAN_LOG_INFO("Using ANO manager {}", mgr);

    // DPDK is the default manager backend
    if (mgr == "default" || mgr == "dpdk") {
      if (rx_en) {
        auto adv_net_rx =
            make_operator<ops::AdvNetworkOpRx>("adv_network_rx",
                                               from_config("advanced_network"),
                                               make_condition<BooleanCondition>("is_alive", true));
        auto bench_rx =
            make_operator<ops::AdvNetworkingBenchDefaultRxOp>("bench_rx", from_config("bench_rx"));
        add_flow(adv_net_rx, bench_rx, {{"bench_rx_out", "burst_in"}});
      }
      if (tx_en) {
        auto adv_net_tx =
            make_operator<ops::AdvNetworkOpTx>("adv_network_tx", from_config("advanced_network"));
        auto bench_tx = make_operator<ops::AdvNetworkingBenchDefaultTxOp>(
            "bench_tx",
            from_config("bench_tx"),
            make_condition<BooleanCondition>("is_alive", true));
        add_flow(bench_tx, adv_net_tx, {{"burst_out", "burst_in"}});
      }
    } else if (mgr == "doca") {
      if (rx_en) {
        auto bench_rx =
            make_operator<ops::AdvNetworkingBenchDocaRxOp>("bench_rx", from_config("bench_rx"));
        auto adv_net_rx =
            make_operator<ops::AdvNetworkOpRx>("adv_network_rx",
                                               from_config("advanced_network"),
                                               make_condition<BooleanCondition>("is_alive", true));
        add_flow(adv_net_rx, bench_rx, {{"bench_rx_out", "burst_in"}});
      }
      if (tx_en) {
        auto bench_tx = make_operator<ops::AdvNetworkingBenchDocaTxOp>(
            "bench_tx",
            from_config("bench_tx"),
            make_condition<BooleanCondition>("is_alive", true));
        auto adv_net_tx =
            make_operator<ops::AdvNetworkOpTx>("adv_network_tx", from_config("advanced_network"));
        add_flow(bench_tx, adv_net_tx, {{"burst_out", "burst_in"}});
      }
    } else {
      HOLOSCAN_LOG_ERROR("Invalid ANO manager/backend");
      exit(1);
    }
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();

  // Get the configuration
  if (argc < 2) {
    HOLOSCAN_LOG_ERROR("Usage: {} config_file", argv[0]);
    return -1;
  }

  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path += "/" + std::string(argv[1]);
  app->config(config_path);
  app->scheduler(app->make_scheduler<holoscan::MultiThreadScheduler>(
      "multithread-scheduler", app->from_config("scheduler")));
  app->run();

  return 0;
}
