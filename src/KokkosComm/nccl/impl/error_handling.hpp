// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once
#include <KokkosComm/gpu/impl/error_handling.hpp>

// Backward-compatible namespace aliases
namespace KokkosComm::nccl {
using KokkosComm::gpu::fail_if;
}  // namespace KokkosComm::nccl
