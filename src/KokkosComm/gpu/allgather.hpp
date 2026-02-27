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
auto allgather(const ExecSpace& space, const SendView& sv, const RecvView& rv, ncclComm_t comm)
    -> Request<GpuCommSpace> {
  using ST = typename SendView::non_const_value_type;
  using RT = typename RecvView::non_const_value_type;
  static_assert(std::is_same_v<ST, RT>,
                "KokkosComm::Experimental::gpu::allgather: View value types must be identical");
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::gpu::allgather");

  Request<GpuCommSpace> req;
  if (KC::is_contiguous(sv) and KC::is_contiguous(rv)) {
    ncclAllGather(KC::data_handle(sv), KC::data_handle(rv), KC::span(sv), datatype<GpuCommSpace, ST>(), comm,
                  KC_GET_STREAM(space));
    req.capture_stream_state(KC_GET_STREAM(space));
  } else {
    Kokkos::abort("KokkosComm::Experimental::gpu::allgather: unimplemented for non-contiguous views");
  }
  req.extend_view_lifetime(sv);
  req.extend_view_lifetime(rv);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace gpu
namespace Impl {

template <KokkosView SendView, KokkosView RecvView>
struct AllGather<SendView, RecvView, KC_GPU_EXEC_SPACE, GpuCommSpace> {
  static auto execute(Handle<KC_GPU_EXEC_SPACE, GpuCommSpace>& h, const SendView sv, RecvView rv)
      -> Request<GpuCommSpace> {
    return gpu::allgather(h.space(), sv, rv, h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm::Experimental
