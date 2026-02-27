// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <cstdio>
#include <string_view>

#include <Kokkos_Core.hpp>

#include <KokkosComm/gpu/gpu_runtime.hpp>

// GPU runtime error check macro
#define KC_GPU_CHECK(expr)                                                                                       \
  ([&]() {                                                                                                       \
    KC_GPU_ERROR_T kcErr = (expr);                                                                               \
    if (KC_GPU_SUCCESS != kcErr) {                                                                               \
      std::fprintf(stderr, "%s:%d: error (" KC_GPU_BACKEND_NAME "): %s\n", __FILE__, __LINE__,                   \
                   KC_GPU_GET_ERROR_STRING(kcErr));                                                               \
    }                                                                                                            \
  }())

// NCCL/RCCL result check macro (shared: both use ncclResult_t + ncclGetErrorString)
#define KC_NCCL_CHECK(expr)                                                                                      \
  ([&]() {                                                                                                       \
    ncclResult_t kcRes = (expr);                                                                                 \
    if (ncclSuccess != kcRes) {                                                                                  \
      std::fprintf(stderr, "%s:%d: error (" KC_GPU_NCCL_BACKEND_NAME "): %s\n", __FILE__, __LINE__,              \
                   ncclGetErrorString(kcRes));                                                                    \
    }                                                                                                            \
  }())

namespace KokkosComm::gpu {

inline auto fail_if(bool condition, std::string_view error_msg) -> void {
  if (condition) {
    std::fprintf(stderr, "error: Kokkos Comm (" KC_GPU_NCCL_BACKEND_NAME ") failed with `%.*s`\n",
                 static_cast<int>(error_msg.size()), error_msg.data());
    Kokkos::abort(error_msg.data());
  }
}

inline auto fail_if(bool condition, std::string_view error_msg, ncclComm_t comm) -> void {
  if (condition) {
#ifdef KOKKOSCOMM_ABORT_ON_ERROR
    std::fprintf(stderr, "error: Kokkos Comm (" KC_GPU_NCCL_BACKEND_NAME ") failed with `%.*s`\n",
                 static_cast<int>(error_msg.size()), error_msg.data());
    ncclCommAbort(comm);
#else
    Kokkos::abort(error_msg.data());
#endif
  }
}

}  // namespace KokkosComm::gpu
