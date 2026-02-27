// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

/// @file gpu_runtime.hpp
/// @brief GPU runtime abstraction layer for NCCL (CUDA) and RCCL (HIP) backends.
///
/// This header provides a set of preprocessor macros that abstract the differences
/// between the CUDA and HIP GPU runtimes, allowing shared implementation code
/// in the gpu/ directory to work with both NCCL and RCCL backends.

#pragma once

#if !defined(KOKKOSCOMM_ENABLE_NCCL) && !defined(KOKKOSCOMM_ENABLE_RCCL)
#error "gpu/gpu_runtime.hpp requires KOKKOSCOMM_ENABLE_NCCL or KOKKOSCOMM_ENABLE_RCCL"
#endif

#include <type_traits>

// ---------------------------------------------------------------------------
// GPU Runtime includes
// ---------------------------------------------------------------------------

#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include <nccl.h>
#include <cuda_runtime.h>
#elif defined(KOKKOSCOMM_ENABLE_RCCL)
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>
#endif

// ---------------------------------------------------------------------------
// GPU Runtime abstraction macros
// ---------------------------------------------------------------------------

#if defined(KOKKOSCOMM_ENABLE_NCCL)

// Stream types and operations
#define KC_GPU_STREAM_T                  cudaStream_t

// Event types and operations
#define KC_GPU_EVENT_T                   cudaEvent_t
#define KC_GPU_EVENT_CREATE_FLAGS(e, f)  cudaEventCreate(&(e), (f))
#define KC_GPU_EVENT_DESTROY(e)          cudaEventDestroy(e)
#define KC_GPU_EVENT_RECORD(e, s)        cudaEventRecord((e), (s))
#define KC_GPU_EVENT_SYNC(e)             cudaEventSynchronize(e)
#define KC_GPU_EVENT_QUERY(e)            cudaEventQuery(e)
#define KC_GPU_EVENT_DISABLE_TIMING      cudaEventDisableTiming

// Error types
#define KC_GPU_ERROR_T                   cudaError_t
#define KC_GPU_SUCCESS                   cudaSuccess
#define KC_GPU_ERROR_NOT_READY           cudaErrorNotReady
#define KC_GPU_GET_ERROR_STRING(e)       cudaGetErrorString(e)

// Backend names for error messages
#define KC_GPU_BACKEND_NAME              "CUDA"
#define KC_GPU_NCCL_BACKEND_NAME         "NCCL"

// Kokkos execution space and stream accessor
#define KC_GPU_EXEC_SPACE                Kokkos::Cuda
#define KC_GET_STREAM(space)             (space).cuda_stream()

#elif defined(KOKKOSCOMM_ENABLE_RCCL)

// Stream types and operations
#define KC_GPU_STREAM_T                  hipStream_t

// Event types and operations
// CRITICAL: cudaEventCreate(&e, flags) -> hipEventCreateWithFlags(&e, flags)
#define KC_GPU_EVENT_T                   hipEvent_t
#define KC_GPU_EVENT_CREATE_FLAGS(e, f)  hipEventCreateWithFlags(&(e), (f))
#define KC_GPU_EVENT_DESTROY(e)          hipEventDestroy(e)
#define KC_GPU_EVENT_RECORD(e, s)        hipEventRecord((e), (s))
#define KC_GPU_EVENT_SYNC(e)             hipEventSynchronize(e)
#define KC_GPU_EVENT_QUERY(e)            hipEventQuery(e)
#define KC_GPU_EVENT_DISABLE_TIMING      hipEventDisableTiming

// Error types
#define KC_GPU_ERROR_T                   hipError_t
#define KC_GPU_SUCCESS                   hipSuccess
#define KC_GPU_ERROR_NOT_READY           hipErrorNotReady
#define KC_GPU_GET_ERROR_STRING(e)       hipGetErrorString(e)

// Backend names for error messages
#define KC_GPU_BACKEND_NAME              "HIP"
#define KC_GPU_NCCL_BACKEND_NAME         "RCCL"

// Kokkos execution space and stream accessor
#define KC_GPU_EXEC_SPACE                Kokkos::HIP
#define KC_GET_STREAM(space)             (space).hip_stream()

#endif

// ---------------------------------------------------------------------------
// Communication space definitions
// ---------------------------------------------------------------------------

#include <KokkosComm/concepts.hpp>

namespace KokkosComm {
namespace Experimental {

#if defined(KOKKOSCOMM_ENABLE_NCCL)
/// The NCCL communication space.
struct NcclSpace {
  using communication_space = NcclSpace;
  using handle_type         = ncclComm_t;
  using request_type        = cudaEvent_t;
  using datatype_type       = ncclDataType_t;
  using reduction_op_type   = ncclRedOp_t;
  using rank_type           = int;
};

using GpuCommSpace = NcclSpace;
#elif defined(KOKKOSCOMM_ENABLE_RCCL)
/// The RCCL communication space.
struct RcclSpace {
  using communication_space = RcclSpace;
  using handle_type         = ncclComm_t;
  using request_type        = hipEvent_t;
  using datatype_type       = ncclDataType_t;
  using reduction_op_type   = ncclRedOp_t;
  using rank_type           = int;
};

using GpuCommSpace = RcclSpace;
#endif

}  // namespace Experimental

#if defined(KOKKOSCOMM_ENABLE_NCCL)
template <>
struct Impl::is_communication_space<Experimental::NcclSpace> : public std::true_type {};
#elif defined(KOKKOSCOMM_ENABLE_RCCL)
template <>
struct Impl::is_communication_space<Experimental::RcclSpace> : public std::true_type {};
#endif

}  // namespace KokkosComm
