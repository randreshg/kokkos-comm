// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include <KokkosComm/gpu/gpu_runtime.hpp>
#include <KokkosComm/gpu/handle.hpp>
#include <KokkosComm/gpu/request.hpp>
#include <KokkosComm/gpu/impl/pack_traits.hpp>

namespace KokkosComm::Experimental {
namespace gpu {

namespace KC = KokkosComm;

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView>
auto alltoall(const ExecSpace& space, const SendView& sv, const RecvView& rv, int count, ncclComm_t comm)
    -> Request<GpuCommSpace> {
  using ST = typename SendView::non_const_value_type;
  using RT = typename RecvView::non_const_value_type;
  static_assert(std::is_same_v<ST, RT>,
                "KokkosComm::Experimental::gpu::alltoall: View value types must be identical");
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::gpu::alltoall");

  Request<GpuCommSpace> req;
  if (KC::is_contiguous(sv) and KC::is_contiguous(rv)) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    ncclAlltoAll(KC::data_handle(sv), KC::data_handle(rv), count, datatype<GpuCommSpace, ST>(), comm,
                 KC_GET_STREAM(space));
#else
    int n_pes;
    ncclCommCount(comm, &n_pes);
    ncclGroupStart();
    for (int r = 0; r < n_pes; ++r) {
      ncclSend(KC::data_handle(sv) + r * count, count, datatype<GpuCommSpace, ST>(), r, comm, KC_GET_STREAM(space));
      ncclRecv(KC::data_handle(rv) + r * count, count, datatype<GpuCommSpace, ST>(), r, comm, KC_GET_STREAM(space));
    }
    ncclGroupEnd();
#endif
    req.capture_stream_state(KC_GET_STREAM(space));
  } else {
    Kokkos::abort("KokkosComm::Experimental::gpu::alltoall: unimplemented for non-contiguous views");
  }
  req.extend_view_lifetime(sv);
  req.extend_view_lifetime(rv);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace gpu
namespace Impl {

template <KokkosView SendView, KokkosView RecvView>
struct AllToAll<SendView, RecvView, KC_GPU_EXEC_SPACE, GpuCommSpace> {
  static auto execute(Handle<KC_GPU_EXEC_SPACE, GpuCommSpace>& h, const SendView sv, RecvView rv, int count)
      -> Request<GpuCommSpace> {
    return gpu::alltoall(h.space(), sv, rv, count, h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm::Experimental
