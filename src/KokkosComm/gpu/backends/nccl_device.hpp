// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

/// @file nccl_device.hpp
/// @brief NCCL/CUDA device backend implementation for KokkosComm.

#pragma once

#include <nccl.h>
#include <cuda_runtime.h>
#include <Kokkos_Core.hpp>

namespace KokkosComm {
namespace Experimental {

// Forward declaration
struct NcclSpace;

}  // namespace Experimental

namespace Gpu {

/// @brief Device traits specialization for NCCL (CUDA) backend.
template <>
struct DeviceTraits<Experimental::NcclSpace> {
  using stream_type = cudaStream_t;
  using event_type = cudaEvent_t;
  using error_type = cudaError_t;
  using exec_space = Kokkos::Cuda;

  static constexpr error_type success = cudaSuccess;
  static constexpr error_type not_ready = cudaErrorNotReady;
  static constexpr const char* backend_name = "CUDA";
  static constexpr const char* comm_backend_name = "NCCL";
  static constexpr unsigned int event_disable_timing = cudaEventDisableTiming;
};

/// @brief Device operations specialization for NCCL (CUDA) backend.
template <>
struct DeviceOps<Experimental::NcclSpace> {
  using traits = DeviceTraits<Experimental::NcclSpace>;

  /// @brief Extract CUDA stream from Kokkos::Cuda execution space.
  static auto get_stream(const typename traits::exec_space& space) -> typename traits::stream_type {
    return space.cuda_stream();
  }

  /// @brief Create CUDA event with specified flags.
  static auto event_create_flags(typename traits::event_type& event, unsigned int flags) -> typename traits::error_type {
    return cudaEventCreate(&event, flags);
  }

  /// @brief Destroy CUDA event.
  static auto event_destroy(typename traits::event_type event) -> typename traits::error_type {
    return cudaEventDestroy(event);
  }

  /// @brief Record CUDA event on stream.
  static auto event_record(typename traits::event_type event, typename traits::stream_type stream) -> typename traits::error_type {
    return cudaEventRecord(event, stream);
  }

  /// @brief Synchronize on CUDA event (blocking).
  static auto event_synchronize(typename traits::event_type event) -> typename traits::error_type {
    return cudaEventSynchronize(event);
  }

  /// @brief Query CUDA event status (non-blocking).
  static auto event_query(typename traits::event_type event) -> typename traits::error_type {
    return cudaEventQuery(event);
  }

  /// @brief Get human-readable error string.
  static auto get_error_string(typename traits::error_type error) -> const char* {
    return cudaGetErrorString(error);
  }
};

}  // namespace Gpu
}  // namespace KokkosComm
