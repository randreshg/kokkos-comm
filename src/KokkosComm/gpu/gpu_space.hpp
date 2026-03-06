// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

/// @file gpu_space.hpp
/// @brief GPU communication space definitions (NcclSpace, RcclSpace).
///
/// This header defines the GPU communication space types using a trait-based
/// abstraction layer for device operations (CUDA/HIP). Follows the same pattern
/// as mpi/mpi_space.hpp but with backend-specific device trait implementations.
///
/// Design Pattern:
/// - DeviceTraits<CommSpace>: Compile-time type mappings (stream, event, error types)
/// - DeviceOps<CommSpace>: Runtime operation wrappers (event ops, error handling)
///
/// Benefits over macro-based approach:
/// - Type safety and IDE support (autocomplete, IntelliSense)
/// - Zero runtime overhead (all template/constexpr)
/// - Extensible (new backends just add specializations)

#pragma once

#if !defined(KOKKOSCOMM_ENABLE_NCCL) && !defined(KOKKOSCOMM_ENABLE_RCCL)
#error "gpu/gpu_space.hpp requires KOKKOSCOMM_ENABLE_NCCL or KOKKOSCOMM_ENABLE_RCCL"
#endif

#include <type_traits>
#include <Kokkos_Core.hpp>
#include <KokkosComm/concepts.hpp>

namespace KokkosComm {
namespace Gpu {

/// @brief Device traits provide compile-time type mappings for GPU backends.
///
/// Each backend (NCCL, RCCL, etc.) specializes this template to provide:
/// - stream_type: GPU stream type (e.g., cudaStream_t, hipStream_t)
/// - event_type: GPU event type (e.g., cudaEvent_t, hipEvent_t)
/// - error_type: GPU error type (e.g., cudaError_t, hipError_t)
/// - exec_space: Kokkos execution space type (e.g., Kokkos::Cuda, Kokkos::HIP)
/// - success: Success error code
/// - not_ready: Error code indicating async operation not yet complete
/// - backend_name: Human-readable backend name for error messages
/// - comm_backend_name: Communication library name (NCCL, RCCL, etc.)
template <typename CommSpace>
struct DeviceTraits;

/// @brief Device operations provide runtime-agnostic wrappers for GPU operations.
///
/// Each backend specializes this template to provide:
/// - get_stream(space): Extract GPU stream from Kokkos execution space
/// - event_create_flags(event, flags): Create event with flags
/// - event_destroy(event): Destroy event
/// - event_record(event, stream): Record event on stream
/// - event_synchronize(event): Block until event completes
/// - event_query(event): Check if event has completed (non-blocking)
/// - get_error_string(error): Get human-readable error message
/// - event_disable_timing: Flag to disable event timing (for lower overhead)
template <typename CommSpace>
struct DeviceOps;

}  // namespace Gpu
}

// Include appropriate backend implementation
#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include <KokkosComm/gpu/backends/nccl_device.hpp>
#elif defined(KOKKOSCOMM_ENABLE_RCCL)
#include <KokkosComm/gpu/backends/rccl_device.hpp>
#endif

namespace KokkosComm {
namespace Experimental {

#if defined(KOKKOSCOMM_ENABLE_NCCL)

/// @brief The NCCL communication space.
///
/// This space provides GPU-accelerated communication using NVIDIA's NCCL library.
struct NcclSpace {
  using communication_space = NcclSpace;
  using handle_type         = ncclComm_t;
  using request_type        = Gpu::DeviceTraits<NcclSpace>::event_type;
  using datatype_type       = ncclDataType_t;
  using reduction_op_type   = ncclRedOp_t;
  using rank_type           = int;

  // Device trait types for backend-agnostic code
  using device_traits = Gpu::DeviceTraits<NcclSpace>;
  using device_ops = Gpu::DeviceOps<NcclSpace>;
};

/// Alias for the active GPU communication space.
using GpuCommSpace = NcclSpace;

#elif defined(KOKKOSCOMM_ENABLE_RCCL)

/// @brief The RCCL communication space.
///
/// This space provides GPU-accelerated communication using AMD's RCCL library.
struct RcclSpace {
  using communication_space = RcclSpace;
  using handle_type         = ncclComm_t;
  using request_type        = Gpu::DeviceTraits<RcclSpace>::event_type;
  using datatype_type       = ncclDataType_t;
  using reduction_op_type   = ncclRedOp_t;
  using rank_type           = int;

  // Device trait types for backend-agnostic code
  using device_traits = Gpu::DeviceTraits<RcclSpace>;
  using device_ops = Gpu::DeviceOps<RcclSpace>;
};

/// Alias for the active GPU communication space.
using GpuCommSpace = RcclSpace;

#endif

}  // namespace Experimental

// Register communication spaces with KokkosComm type system
#if defined(KOKKOSCOMM_ENABLE_NCCL)
template <>
struct Impl::is_communication_space<Experimental::NcclSpace> : public std::true_type {};
#elif defined(KOKKOSCOMM_ENABLE_RCCL)
template <>
struct Impl::is_communication_space<Experimental::RcclSpace> : public std::true_type {};
#endif

}  // namespace KokkosComm
