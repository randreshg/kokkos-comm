// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <KokkosComm/config.hpp>
#include "concepts.hpp"
#include "datatype.hpp"
#include "reduction_op.hpp"

namespace KokkosComm {

#if defined(KOKKOSCOMM_ENABLE_RCCL) || defined(KOKKOSCOMM_ENABLE_NCCL)
// GPU backend (NCCL or RCCL) is the default; MPI is fallback
struct MpiSpace;

using DefaultCommunicationSpace  = Experimental::GpuCommSpace;
using FallbackCommunicationSpace = MpiSpace;
#elif defined(KOKKOSCOMM_ENABLE_MPI)
struct MpiSpace;
using DefaultCommunicationSpace  = MpiSpace;
using FallbackCommunicationSpace = MpiSpace;
#else
#error at least one communication space must be enabled
#endif

/// @brief Template class for request wrappers.
template <CommunicationSpace CommSpace = DefaultCommunicationSpace>
class Request;

template <KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace   = DefaultCommunicationSpace>
class Handle;

namespace Impl {

template <KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
struct Recv;

template <KokkosView SendView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
struct Send;

}  // namespace Impl

// Collectives are currently experimental functions
namespace Experimental::Impl {

template <KokkosView View, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
struct Broadcast;

template <KokkosView SendView, KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
struct AllGather;

template <KokkosView SendView, KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
struct AllToAll;

template <KokkosView SendView, KokkosView RecvView, ReductionOperator RedOp,
          KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace   = DefaultCommunicationSpace>
struct AllReduce;

template <KokkosView SendView, KokkosView RecvView, ReductionOperator RedOp,
          KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace   = DefaultCommunicationSpace>
struct Reduce;

}  // namespace Experimental::Impl

}  // namespace KokkosComm
