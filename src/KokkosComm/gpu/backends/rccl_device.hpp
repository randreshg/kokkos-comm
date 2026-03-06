// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

/// @file rccl_device.hpp
/// @brief RCCL/HIP device backend implementation for KokkosComm.

#pragma once

#include <rccl/rccl.h>
#include <hip/hip_runtime.h>
#include <Kokkos_Core.hpp>

namespace KokkosComm {
namespace Experimental {

// Forward declaration
struct RcclSpace;

}  // namespace Experimental

namespace Gpu {

/// @brief Device traits specialization for RCCL (HIP) backend.
template <>
struct DeviceTraits<Experimental::RcclSpace> {
  using stream_type = hipStream_t;
  using event_type = hipEvent_t;
  using error_type = hipError_t;
  using exec_space = Kokkos::HIP;

  static constexpr error_type success = hipSuccess;
  static constexpr error_type not_ready = hipErrorNotReady;
  static constexpr const char* backend_name = "HIP";
  static constexpr const char* comm_backend_name = "RCCL";
  static constexpr unsigned int event_disable_timing = hipEventDisableTiming;
};

/// @brief Device operations specialization for RCCL (HIP) backend.
template <>
struct DeviceOps<Experimental::RcclSpace> {
  using traits = DeviceTraits<Experimental::RcclSpace>;

  /// @brief Extract HIP stream from Kokkos::HIP execution space.
  static auto get_stream(const typename traits::exec_space& space) -> typename traits::stream_type {
    return space.hip_stream();
  }

  /// @brief Create HIP event with specified flags.
  /// NOTE: HIP uses hipEventCreateWithFlags instead of cudaEventCreate.
  static auto event_create_flags(typename traits::event_type& event, unsigned int flags) -> typename traits::error_type {
    return hipEventCreateWithFlags(&event, flags);
  }

  /// @brief Destroy HIP event.
  static auto event_destroy(typename traits::event_type event) -> typename traits::error_type {
    return hipEventDestroy(event);
  }

  /// @brief Record HIP event on stream.
  static auto event_record(typename traits::event_type event, typename traits::stream_type stream) -> typename traits::error_type {
    return hipEventRecord(event, stream);
  }

  /// @brief Synchronize on HIP event (blocking).
  static auto event_synchronize(typename traits::event_type event) -> typename traits::error_type {
    return hipEventSynchronize(event);
  }

  /// @brief Query HIP event status (non-blocking).
  static auto event_query(typename traits::event_type event) -> typename traits::error_type {
    return hipEventQuery(event);
  }

  /// @brief Get human-readable error string.
  static auto get_error_string(typename traits::error_type error) -> const char* {
    return hipGetErrorString(error);
  }
};

}  // namespace Gpu
}  // namespace KokkosComm
