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

template <KokkosExecutionSpace ExecSpace, KokkosView SendView>
auto send(const ExecSpace& space, const SendView& sv, int peer, ncclComm_t comm) -> Request<GpuCommSpace> {
  using T = typename SendView::non_const_value_type;
  Kokkos::Tools::pushRegion("KokkosComm::Impl::send");

  Request<GpuCommSpace> req;
  if (is_contiguous(sv)) {
    KC_NCCL_CHECK(ncclSend(data_handle(sv), span(sv), datatype<GpuCommSpace, T>(), peer, comm, KC_GET_STREAM(space)));
    req.capture_stream_state(KC_GET_STREAM(space));
  } else {
    using Packer = typename Impl::PackTraits<SendView>::packer_type;
    auto pckd_sv = Packer::pack(space, "pckd_sv", sv);
    KC_NCCL_CHECK(
        ncclSend(data_handle(pckd_sv.view_), pckd_sv.count_, pckd_sv.datatype_, peer, comm, KC_GET_STREAM(space)));
    req.capture_stream_state(KC_GET_STREAM(space));
    req.extend_view_lifetime(pckd_sv.view_);
  }
  req.extend_view_lifetime(sv);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace Experimental::gpu
namespace Impl {

template <KokkosView SendView>
struct Send<SendView, KC_GPU_EXEC_SPACE, Experimental::GpuCommSpace> {
  static auto execute(Handle<KC_GPU_EXEC_SPACE, Experimental::GpuCommSpace>& h, SendView sv, int peer)
      -> Request<Experimental::GpuCommSpace> {
    return Experimental::gpu::send(h.space(), sv, peer, h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm
