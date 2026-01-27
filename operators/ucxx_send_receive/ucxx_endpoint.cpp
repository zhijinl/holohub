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

#include "ucxx_endpoint.hpp"

#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <vector>

namespace holoscan::ops {

void UcxxEndpoint::add_close_callback(std::function<void(ucs_status_t)> callback) {
  std::scoped_lock lock(close_callbacks_mutex_);
  close_callbacks_.push_back(std::move(callback));
}

UcxxEndpoint::~UcxxEndpoint() {
  stop_listen_.store(true, std::memory_order_release);
  if (listen_fd_ >= 0) {
    ::shutdown(listen_fd_, SHUT_RDWR);
    ::close(listen_fd_);
    listen_fd_ = -1;
  }
  if (listen_thread_.joinable()) {
    listen_thread_.join();
  }
  if (worker_) {
    worker_->stopProgressThread();
  }
}

void UcxxEndpoint::setup(holoscan::ComponentSpec& spec) {
  spec.param(hostname_, "hostname", "Hostname", "Hostname of the endpoint",
             std::string("127.0.0.1"));
  spec.param(port_, "port", "Port", "Port of the endpoint", 50008);
  spec.param(listen_, "listen", "Listen",
             "Whether to listen for connections (server), or initiate a connection (client)",
             false);

  is_alive_condition_ = fragment()->make_condition<holoscan::AsynchronousCondition>(
      fmt::format("{}_is_alive", name()));
}

namespace {

int create_listen_socket(const std::string& hostname, int port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) { return -1; }

  int opt = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));

  if (hostname.empty() || hostname == "0.0.0.0") {
    addr.sin_addr.s_addr = INADDR_ANY;
  } else {
    if (::inet_pton(AF_INET, hostname.c_str(), &addr.sin_addr) != 1) {
      ::close(fd);
      return -1;
    }
  }

  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(fd);
    return -1;
  }
  if (::listen(fd, 8) != 0) {
    ::close(fd);
    return -1;
  }
  return fd;
}

int connect_socket(const std::string& hostname, int port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) { return -1; }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (::inet_pton(AF_INET, hostname.c_str(), &addr.sin_addr) != 1) {
    ::close(fd);
    return -1;
  }

  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(fd);
    return -1;
  }
  return fd;
}

bool send_all(int fd, const void* data, size_t size) {
  const auto* buffer = static_cast<const uint8_t*>(data);
  size_t sent = 0;
  while (sent < size) {
    const ssize_t n = ::send(fd, buffer + sent, size - sent, 0);
    if (n <= 0) { return false; }
    sent += static_cast<size_t>(n);
  }
  return true;
}

bool recv_all(int fd, void* data, size_t size) {
  auto* buffer = static_cast<uint8_t*>(data);
  size_t received = 0;
  while (received < size) {
    const ssize_t n = ::recv(fd, buffer + received, size - received, 0);
    if (n <= 0) { return false; }
    received += static_cast<size_t>(n);
  }
  return true;
}

}  // namespace

void UcxxEndpoint::initialize() {
  if (is_initialized_) {
    return;
  }
  holoscan::Resource::initialize();

  context_ = ::ucxx::createContext({}, ::ucxx::Context::defaultFeatureFlags);
  worker_ = context_->createWorker();

  is_alive_condition_->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);

  if (listen_) {
    listen_fd_ = create_listen_socket(hostname_.get(), port_.get());
    if (listen_fd_ < 0) {
      HOLOSCAN_LOG_ERROR("Failed to open control socket on {}:{} (errno: {})",
                         hostname_.get(), port_.get(), errno);
      return;
    }

    HOLOSCAN_LOG_INFO("Listening on: {}", port_.get());
    listen_thread_ = std::thread([this]() {
      while (!stop_listen_.load(std::memory_order_acquire)) {
        int client_fd = ::accept(listen_fd_, nullptr, nullptr);
        if (client_fd < 0) {
          if (stop_listen_.load(std::memory_order_acquire)) { break; }
          continue;
        }

        auto local_address = worker_->getAddress();
        const std::string local_address_str = local_address->getString();

        uint64_t remote_len = 0;
        if (!recv_all(client_fd, &remote_len, sizeof(remote_len))) {
          ::close(client_fd);
          continue;
        }
        std::string remote_address_str(remote_len, '\0');
        if (!recv_all(client_fd, remote_address_str.data(), remote_address_str.size())) {
          ::close(client_fd);
          continue;
        }
        const uint64_t local_len_u64 = static_cast<uint64_t>(local_address_str.size());
        if (!send_all(client_fd, &local_len_u64, sizeof(local_len_u64)) ||
            !send_all(client_fd, local_address_str.data(), local_address_str.size())) {
          ::close(client_fd);
          continue;
        }

        auto remote_address = ::ucxx::createAddressFromString(remote_address_str);
        auto ep = worker_->createEndpointFromWorkerAddress(remote_address, true);
        std::atomic_store(&endpoint_, ep);

        HOLOSCAN_LOG_INFO("Endpoint connected");
        is_alive_condition_->event_state(holoscan::AsynchronousEventState::EVENT_DONE);

        ep->setCloseCallback(
            [this](ucs_status_t status, std::shared_ptr<void>) { on_endpoint_closed(status); },
            nullptr);

        ::close(client_fd);
      }
    });
  } else {
    int fd = connect_socket(hostname_.get(), port_.get());
    if (fd < 0) {
      HOLOSCAN_LOG_ERROR("Failed to connect control socket to {}:{} (errno: {})",
                         hostname_.get(), port_.get(), errno);
      return;
    }

    auto local_address = worker_->getAddress();
    const std::string local_address_str = local_address->getString();

    const uint64_t local_len_u64 = static_cast<uint64_t>(local_address_str.size());
    if (!send_all(fd, &local_len_u64, sizeof(local_len_u64)) ||
        !send_all(fd, local_address_str.data(), local_address_str.size())) {
      ::close(fd);
      HOLOSCAN_LOG_ERROR("Failed to send worker address to {}:{}", hostname_.get(), port_.get());
      return;
    }

    uint64_t remote_len = 0;
    if (!recv_all(fd, &remote_len, sizeof(remote_len))) {
      ::close(fd);
      HOLOSCAN_LOG_ERROR("Failed to receive worker address length from {}:{}",
                         hostname_.get(), port_.get());
      return;
    }
    std::string remote_address_str(remote_len, '\0');
    if (!recv_all(fd, remote_address_str.data(), remote_address_str.size())) {
      ::close(fd);
      HOLOSCAN_LOG_ERROR("Failed to receive worker address from {}:{}", hostname_.get(), port_.get());
      return;
    }
    ::close(fd);

    auto remote_address = ::ucxx::createAddressFromString(remote_address_str);
    auto ep = worker_->createEndpointFromWorkerAddress(remote_address, true);
    std::atomic_store(&endpoint_, ep);

    HOLOSCAN_LOG_INFO("Connecting to: {}:{}", hostname_.get(), port_.get());

    // Mark operators ready to execute.
    is_alive_condition_->event_state(holoscan::AsynchronousEventState::EVENT_DONE);

    ep->setCloseCallback(
        [this](ucs_status_t status, std::shared_ptr<void>) { on_endpoint_closed(status); },
        nullptr);
  }

  worker_->startProgressThread(/*pollingMode=*/false);
}

void UcxxEndpoint::on_endpoint_closed(ucs_status_t status) {
  HOLOSCAN_LOG_INFO("Endpoint closed");
  if (status != UCS_OK) {
    // These are expected when subscriber disconnects/restarts.
    if (status == UCS_ERR_CONNECTION_RESET || status == UCS_ERR_NOT_CONNECTED ||
        status == UCS_ERR_UNREACHABLE || status == UCS_ERR_CANCELED) {
      HOLOSCAN_LOG_WARN("Endpoint closed (likely disconnect/reconnect) with status: {}",
                        ucs_status_string(status));
    } else {
      HOLOSCAN_LOG_ERROR("Endpoint closed with status: {}", ucs_status_string(status));
    }
  }

  // Notify any registered callbacks. (May be invoked from UCXX progress thread.)
  {
    std::vector<std::function<void(ucs_status_t)>> callbacks_copy;
    {
      std::scoped_lock lock(close_callbacks_mutex_);
      callbacks_copy = close_callbacks_;
    }
    for (auto& cb : callbacks_copy) {
      if (cb) { cb(status); }
    }
  }

  // Clear the endpoint so operators can quickly detect disconnection.
  std::atomic_store(&endpoint_, std::shared_ptr<::ucxx::Endpoint>{});

  // Prevent operators from executing until a new connection is established (server mode)
  // or indefinitely (client mode).
  if (listen_) {
    is_alive_condition_->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);
  } else {
    is_alive_condition_->event_state(holoscan::AsynchronousEventState::EVENT_NEVER);
  }
}

};  // namespace holoscan::ops
