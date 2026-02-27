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
#include <KokkosComm/gpu/impl/error_handling.hpp>

namespace KokkosComm {
namespace Experimental::gpu {

template <KokkosExecutionSpace ExecSpace, KokkosView RecvView>
auto recv(const ExecSpace& space, RecvView& rv, int peer, ncclComm_t comm) -> Request<GpuCommSpace> {
  using T = typename RecvView::non_const_value_type;
  Kokkos::Tools::pushRegion("KokkosComm::Impl::recv");

  Request<GpuCommSpace> req;
  if (is_contiguous(rv)) {
    KC_NCCL_CHECK(ncclRecv(data_handle(rv), span(rv), datatype<GpuCommSpace, T>(), peer, comm, KC_GET_STREAM(space)));
    req.capture_stream_state(KC_GET_STREAM(space));
  } else {
    using Packer = typename Impl::PackTraits<RecvView>::packer_type;
    auto pckd_rv = Packer::allocate_packed_for(space, "pckd_rv", rv);
    KC_NCCL_CHECK(
        ncclRecv(data_handle(pckd_rv.view_), pckd_rv.count_, pckd_rv.datatype_, peer, comm, KC_GET_STREAM(space)));
    req.capture_stream_state(KC_GET_STREAM(space));
    req.add_callback([space, rv, pckd_rv]() {
      Packer::unpack_into(space, rv, pckd_rv.view_);
      space.fence("fence `pckd_rv` unpacking after " KC_GPU_NCCL_BACKEND_NAME " call");
    });
  }
  req.extend_view_lifetime(rv);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace Experimental::gpu
namespace Impl {

template <KokkosView RecvView>
struct Recv<RecvView, KC_GPU_EXEC_SPACE, Experimental::GpuCommSpace> {
  static auto execute(Handle<KC_GPU_EXEC_SPACE, Experimental::GpuCommSpace>& h, RecvView sv, int peer)
      -> Request<Experimental::GpuCommSpace> {
    return Experimental::gpu::recv(h.space(), sv, peer, h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm
