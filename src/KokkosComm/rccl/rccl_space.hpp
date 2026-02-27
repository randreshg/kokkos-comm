// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

/// @file rccl_space.hpp
/// @brief Thin wrapper that includes the shared GPU communication space definition.
/// When KOKKOSCOMM_ENABLE_RCCL is defined, GpuCommSpace resolves to RcclSpace.

#pragma once
#include <KokkosComm/gpu/gpu_runtime.hpp>
