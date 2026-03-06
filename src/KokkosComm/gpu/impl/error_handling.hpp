// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <cstdio>
#include <string_view>

#include <Kokkos_Core.hpp>

#include <KokkosComm/gpu/gpu_space.hpp>

// Device-agnostic GPU runtime error check macro
// Uses the device_ops trait to get the error string
#define KC_GPU_CHECK(expr)                                                                                       \
  ([&]() {                                                                                                       \
    auto kcErr = (expr);                                                                                         \
    using CommSpace = KokkosComm::Experimental::GpuCommSpace;                                                    \
    using DeviceTraits = typename CommSpace::device_traits;                                                      \
    using DeviceOps = typename CommSpace::device_ops;                                                            \
    if (DeviceTraits::success != kcErr) {                                                                        \
      std::fprintf(stderr, "%s:%d: error (%s): %s\n", __FILE__, __LINE__,                                        \
                   DeviceTraits::backend_name, DeviceOps::get_error_string(kcErr));                              \
    }                                                                                                            \
  }())

// NCCL/RCCL result check macro (shared: both use ncclResult_t + ncclGetErrorString)
#define KC_NCCL_CHECK(expr)                                                                                      \
  ([&]() {                                                                                                       \
    ncclResult_t kcRes = (expr);                                                                                 \
    using CommSpace = KokkosComm::Experimental::GpuCommSpace;                                                    \
    using DeviceTraits = typename CommSpace::device_traits;                                                      \
    if (ncclSuccess != kcRes) {                                                                                  \
      std::fprintf(stderr, "%s:%d: error (%s): %s\n", __FILE__, __LINE__,                                        \
                   DeviceTraits::comm_backend_name, ncclGetErrorString(kcRes));                                  \
    }                                                                                                            \
  }())

namespace KokkosComm::gpu {

inline auto fail_if(bool condition, std::string_view error_msg) -> void {
  using CommSpace = KokkosComm::Experimental::GpuCommSpace;
  using DeviceTraits = typename CommSpace::device_traits;

  if (condition) {
    std::fprintf(stderr, "error: Kokkos Comm (%s) failed with `%.*s`\n",
                 DeviceTraits::comm_backend_name,
                 static_cast<int>(error_msg.size()), error_msg.data());
    Kokkos::abort(error_msg.data());
  }
}

inline auto fail_if(bool condition, std::string_view error_msg, ncclComm_t comm) -> void {
  using CommSpace = KokkosComm::Experimental::GpuCommSpace;
  using DeviceTraits = typename CommSpace::device_traits;

  if (condition) {
#ifdef KOKKOSCOMM_ABORT_ON_ERROR
    std::fprintf(stderr, "error: Kokkos Comm (%s) failed with `%.*s`\n",
                 DeviceTraits::comm_backend_name,
                 static_cast<int>(error_msg.size()), error_msg.data());
    ncclCommAbort(comm);
#else
    Kokkos::abort(error_msg.data());
#endif
  }
}

}  // namespace KokkosComm::gpu
