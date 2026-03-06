// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

/// @file sendrecv.hpp
/// @brief Grouped send+recv operation for GPU backends.
///
/// Provides a sendrecv helper that automatically wraps concurrent send and recv
/// operations in NCCL/RCCL group semantics (ncclGroupStart/ncclGroupEnd).
///
/// This is critical for RCCL to prevent deadlocks when operations are issued
/// concurrently. While individual send() and recv() calls work fine when called
/// separately, concurrent send+recv patterns (like quantum circuit exchanges in
/// PennyLane) require proper grouping.
///
/// Usage:
///   auto [send_req, recv_req] = sendrecv(space, send_view, dest_rank,
///                                         recv_view, source_rank, comm);
///   wait(send_req);
///   wait(recv_req);

#pragma once

#include <utility>
#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/gpu/send.hpp>
#include <KokkosComm/gpu/recv.hpp>
#include <KokkosComm/gpu/impl/group_semantics.hpp>

namespace KokkosComm {
namespace Experimental::gpu {

/// @brief Perform concurrent send and recv operations with proper NCCL/RCCL grouping.
///
/// This function wraps send and recv operations in ncclGroupStart/ncclGroupEnd
/// to enable concurrent execution without deadlock (critical for RCCL).
///
/// @tparam ExecSpace Kokkos execution space type
/// @tparam SendView Kokkos view type for send buffer
/// @tparam RecvView Kokkos view type for receive buffer
/// @param space Execution space instance
/// @param sv Send view
/// @param dest Destination rank
/// @param rv Receive view
/// @param source Source rank
/// @param comm NCCL/RCCL communicator
/// @return Pair of requests (send_request, recv_request)
template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView>
auto sendrecv(const ExecSpace& space,
              const SendView& sv, int dest,
              RecvView& rv, int source,
              ncclComm_t comm) -> std::pair<Request<GpuCommSpace>, Request<GpuCommSpace>> {

  Kokkos::Tools::pushRegion("KokkosComm::Experimental::gpu::sendrecv");

  // Group send and recv to prevent deadlock (critical for RCCL)
  Gpu::GroupScope group;

  auto send_req = send(space, sv, dest, comm);
  auto recv_req = recv(space, rv, source, comm);

  Kokkos::Tools::popRegion();

  return {std::move(send_req), std::move(recv_req)};
}

}  // namespace Experimental::gpu
}  // namespace KokkosComm
