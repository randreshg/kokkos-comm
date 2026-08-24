// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <functional>
#include <utility>
#include <memory>
#include <span>
#include <vector>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/fwd.hpp>
#include <KokkosComm/gpu/gpu_space.hpp>
#include <KokkosComm/gpu/impl/error_handling.hpp>

namespace KokkosComm {

/// @brief Request specialization for the GPU communication space (NCCL or RCCL).
template <>
class Request<Experimental::GpuCommSpace> {
 public:
  using communication_space = Experimental::GpuCommSpace;
  using request_type        = Experimental::GpuCommSpace::request_type;
  using rank_type           = Experimental::GpuCommSpace::rank_type;

  /// @brief Constructs a `Request`.
  explicit Request() : request_(nullptr) {}

  /// @brief Capture the state of a GPU stream for request encapsulation.
  /// @param stream The stream to capture for request encapsulation.
  auto capture_stream_state(typename communication_space::device_traits::stream_type stream) noexcept -> void {
    using device_ops = typename communication_space::device_ops;
    using device_traits = typename communication_space::device_traits;

    if (request_ != nullptr) {
      KC_GPU_CHECK(device_ops::event_destroy(request_));
    }
    KC_GPU_CHECK(device_ops::event_create_flags(request_, device_traits::event_disable_timing));
    KC_GPU_CHECK(device_ops::event_record(request_, stream));
  }

  /// @brief Destructor.
  ~Request() noexcept {
    using device_ops = typename communication_space::device_ops;

    if (request_ != nullptr) {
      KC_GPU_CHECK(device_ops::event_destroy(request_));
    }
  };

  /// @brief Copy constructor is deleted because a `Request` can only be moved.
  Request(const Request&) = delete;
  /// @brief Copy assignment operator is deleted because a `Request` can only be moved.
  auto operator=(const Request&) -> Request& = delete;
  /// @brief Move constructor.
  /// The defaulted move copied `request_` verbatim, leaving both the source and
  /// the destination owning the same GPU event. The moved-from object then
  /// destroyed that event in its destructor while the destination was still
  /// holding it, giving a double free / use-after-free on the event handle.
  Request(Request&& other) noexcept : request_(other.request_), callbacks_(std::move(other.callbacks_)) {
    other.request_ = nullptr;
  }
  /// @brief Move assignment operator.
  auto operator=(Request&& other) noexcept -> Request& {
    if (this != &other) {
      using device_ops = typename communication_space::device_ops;
      if (request_ != nullptr) {
        KC_GPU_CHECK(device_ops::event_destroy(request_));
      }
      request_       = other.request_;
      other.request_ = nullptr;
      callbacks_     = std::move(other.callbacks_);
    }
    return *this;
  }

  /// @return A reference to the underlying GPU event object.
  [[nodiscard]] constexpr auto request() noexcept -> request_type& { return request_; }
  /// @return A const reference to the underlying GPU event object.
  [[nodiscard]] constexpr auto request() const noexcept -> const request_type& { return request_; }
  /// @return A pointer to the underlying GPU event object.
  [[nodiscard]] constexpr auto request_ptr() noexcept -> request_type* { return &request_; }
  /// @return A const pointer to the underlying GPU event object.
  [[nodiscard]] constexpr auto request_ptr() const noexcept -> const request_type* { return &request_; }

  /// @brief Adds a function to a list of callbacks to be invoked after the request's completion.
  /// @param cb The callback function to register.
  auto add_callback(std::function<void()>&& cb) -> void { callbacks_.push_back(cb); }

  /// @brief Captures a Kokkos View to extend its lifetime until the request's completion.
  /// @tparam V A Kokkos View type.
  /// @param view The Kokkos View to capture for lifetime extension.
  template <KokkosView V>
  auto extend_view_lifetime(const V& view) -> void {
    // Unmanaged views don't own the underlying buffer, so no need to extend their lifetime
    if (view.use_count() != 0) {
      add_callback([view]() {});
    }
  }

  /// @brief Waits on the request until completion of the associated operation.
  auto wait() -> void {
    using device_ops = typename communication_space::device_ops;
    using device_traits = typename communication_space::device_traits;

    auto err = device_ops::event_synchronize(request_);
    gpu::fail_if(err != device_traits::success, "KokkosComm::Request::wait: request completion failed");

    execute_all_callbacks();
  }

  /// @brief Queries the request for the completion of the associated operation.
  /// If the operation has completed, all callbacks are executed upon return, similarly to having called `wait`.
  /// @return True if the request has completed or is null/inactive, false otherwise.
  [[nodiscard]] auto test() -> bool {
    using device_ops = typename communication_space::device_ops;
    using device_traits = typename communication_space::device_traits;

    auto err = device_ops::event_query(request_);
    if (err == device_traits::success) {
      execute_all_callbacks();
      return true;
    } else if (err == device_traits::not_ready) {
      return false;
    }

    gpu::fail_if(err != device_traits::success, "KokkosComm::Request::test: request completion failed");
    return false;  // unreachable
  }

 private:
  request_type request_;
  std::vector<std::function<void()>> callbacks_;

  /// @brief Executes all the callbacks registered on the request.
  auto execute_all_callbacks() -> void {
    for (auto& cb : callbacks_) {
      cb();
    }
    callbacks_.clear();
  }

  friend auto wait(Request<communication_space>& request) -> void;
  friend auto wait(Request<communication_space>&& request) -> void;
  friend auto wait_all(std::span<Request<communication_space>> requests) -> void;
  friend auto wait_any(std::span<Request<communication_space>> requests) -> std::optional<rank_type>;
  friend auto test(Request<communication_space>& request) -> bool;
};

/// @brief Waits on the request until completion of the associated operation.
/// @param request A reference on the request to wait for completion.
inline auto wait(Request<Experimental::GpuCommSpace>& request) -> void { request.wait(); }
/// @brief Waits on the request until completion of the associated operation.
/// @param request An r-value reference on the request, consumed upon completion.
inline auto wait(Request<Experimental::GpuCommSpace>&& request) -> void { request.wait(); }

/// @brief Waits for completion of all passed requests.
/// @param requests The list of requests to complete.
inline auto wait_all(std::span<Request<Experimental::GpuCommSpace>> requests) -> void {
  using device_ops = typename Experimental::GpuCommSpace::device_ops;
  using device_traits = typename Experimental::GpuCommSpace::device_traits;

  if (requests.empty()) {
    return;
  }

  int remaining = requests.size();
  // Poll until all requests are completed
  while (remaining > 0) {
    for (auto& req : requests) {
      auto err = device_ops::event_query(req.request());
      if (err == device_traits::success) {
        req.execute_all_callbacks();
        remaining--;
      } else if (err == device_traits::not_ready) {
        continue;
      } else {
        gpu::fail_if(err != device_traits::success, "KokkosComm::Request::wait_all: request completions failed");
      }
    }
  }
}

/// @brief Waits for the completion of one request among all passed requests.
/// @param requests The list of requests to try to complete.
/// @return The index of the request within the passed list upon successful completion, `std::nullopt` otherwise.
inline auto wait_any(std::span<Request<Experimental::GpuCommSpace>> requests)
    -> std::optional<typename Request<Experimental::GpuCommSpace>::rank_type> {
  using device_ops = typename Experimental::GpuCommSpace::device_ops;
  using device_traits = typename Experimental::GpuCommSpace::device_traits;

  if (requests.empty()) {
    return std::nullopt;
  }

  while (true) {
    for (size_t r = 0; r < requests.size(); ++r) {
      auto err = device_ops::event_query(requests[r].request());
      if (err == device_traits::success) {
        requests[r].execute_all_callbacks();
        return static_cast<typename Request<Experimental::GpuCommSpace>::rank_type>(r);
      } else if (err == device_traits::not_ready) {
        continue;
      } else {
        gpu::fail_if(err != device_traits::success, "KokkosComm::Request::wait_any: request completion failed");
      }
    }
  }
}

/// @brief Queries the request for completion of the associated operation.
/// @param request A reference on the request to query its completion.
inline auto test(Request<Experimental::GpuCommSpace>& request) -> bool { return request.test(); }

}  // namespace KokkosComm
