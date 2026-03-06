// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

/// @file group_semantics.hpp
/// @brief NCCL/RCCL group semantics support for concurrent operations.
///
/// RAII wrapper for ncclGroupStart/ncclGroupEnd to ensure proper grouping
/// of concurrent send/recv operations. This is critical for RCCL to prevent
/// deadlocks when operations are issued concurrently.
///
/// Background:
/// NCCL/RCCL group semantics allow multiple communication operations to be
/// batched and fused for better performance. For RCCL specifically, grouping
/// concurrent send/recv operations is required to avoid hangs.
///
/// Usage:
///   {
///     Gpu::GroupScope group(comm);
///     auto req1 = send(...);
///     auto req2 = recv(...);
///   }  // ncclGroupEnd called automatically

#pragma once

#include <nccl.h>
#include <KokkosComm/gpu/impl/error_handling.hpp>

namespace KokkosComm {
namespace Gpu {

/// @brief RAII wrapper for NCCL/RCCL group semantics.
///
/// Automatically calls ncclGroupStart() on construction and ncclGroupEnd()
/// on destruction, ensuring proper cleanup even in the presence of exceptions.
class GroupScope {
 public:
  /// @brief Start NCCL/RCCL group.
  explicit GroupScope() {
    KC_NCCL_CHECK(ncclGroupStart());
  }

  /// @brief End NCCL/RCCL group.
  ~GroupScope() {
    KC_NCCL_CHECK(ncclGroupEnd());
  }

  // Non-copyable, non-movable
  GroupScope(const GroupScope&) = delete;
  GroupScope& operator=(const GroupScope&) = delete;
  GroupScope(GroupScope&&) = delete;
  GroupScope& operator=(GroupScope&&) = delete;
};

}  // namespace Gpu
}  // namespace KokkosComm
