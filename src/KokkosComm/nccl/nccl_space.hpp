// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

/// @file nccl_space.hpp
/// @brief Thin wrapper that includes the shared GPU communication space definition.
/// When KOKKOSCOMM_ENABLE_NCCL is defined, GpuCommSpace resolves to NcclSpace.

#pragma once
#include <KokkosComm/gpu/gpu_runtime.hpp>
