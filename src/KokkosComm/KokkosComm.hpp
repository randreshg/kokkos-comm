// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include "fwd.hpp"
#include "concepts.hpp"
#include "point_to_point.hpp"
#include "collective.hpp"

// Communication spaces declarations
#if defined(KOKKOSCOMM_ENABLE_MPI)
#include "mpi/mpi_space.hpp"

#include "mpi/channel.hpp"
#include "mpi/comm_mode.hpp"
#include "mpi/handle.hpp"
#include "mpi/request.hpp"

#include "mpi/irecv.hpp"
#include "mpi/isend.hpp"
#include "mpi/recv.hpp"
#include "mpi/send.hpp"

#include "mpi/broadcast.hpp"
#include "mpi/allgather.hpp"
#include "mpi/alltoall.hpp"
#include "mpi/allreduce.hpp"
#include "mpi/reduce.hpp"
#include "mpi/scan.hpp"

#include "mpi/barrier.hpp"
#endif

#if defined(KOKKOSCOMM_ENABLE_NCCL) || defined(KOKKOSCOMM_ENABLE_RCCL)
#include "gpu/gpu_space.hpp"

#include "gpu/handle.hpp"
#include "gpu/request.hpp"

#include "gpu/recv.hpp"
#include "gpu/send.hpp"

#include "gpu/broadcast.hpp"
#include "gpu/allgather.hpp"
#include "gpu/alltoall.hpp"
#include "gpu/allreduce.hpp"
#include "gpu/reduce.hpp"
#endif

#if !defined(KOKKOSCOMM_ENABLE_MPI) && !defined(KOKKOSCOMM_ENABLE_NCCL) && !defined(KOKKOSCOMM_ENABLE_RCCL)
static_assert(false, "KokkosComm: at least one communication space must be defined");
#endif

namespace KokkosComm {}  // namespace KokkosComm
