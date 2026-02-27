// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <type_traits>

#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include <KokkosComm/reduction_op.hpp>
#include <KokkosComm/gpu/gpu_runtime.hpp>
#include <KokkosComm/gpu/handle.hpp>
#include <KokkosComm/gpu/request.hpp>
#include <KokkosComm/gpu/impl/pack_traits.hpp>

namespace KokkosComm::Experimental {
namespace gpu {

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView>
auto reduce(const ExecSpace& space, const SendView& sv, RecvView& rv, ncclRedOp_t op, int root, int rank,
            ncclComm_t comm) -> Request<GpuCommSpace> {
  using ST         = typename SendView::non_const_value_type;
  using RT         = typename RecvView::non_const_value_type;
  using SendPacker = typename Impl::PackTraits<SendView>::packer_type;
  using RecvPacker = typename Impl::PackTraits<RecvView>::packer_type;
  static_assert(std::is_same_v<ST, RT>, "KokkosComm::Experimental::gpu::reduce: View value types must be identical");
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::gpu::reduce");

  Request<GpuCommSpace> req;
  if (is_contiguous(sv)) {
    if (rank != root and is_contiguous(rv)) {
      ncclReduce(data_handle(sv), data_handle(rv), span(sv), datatype<GpuCommSpace, ST>(), op, root, comm,
                 KC_GET_STREAM(space));
      req.capture_stream_state(KC_GET_STREAM(space));
    } else {
      auto pckd_rv = RecvPacker::allocate_packed_for(space, "pckd_rv", rv);
      ncclReduce(data_handle(sv), data_handle(pckd_rv.view_), span(sv), datatype<GpuCommSpace, ST>(), op, root, comm,
                 KC_GET_STREAM(space));
      req.capture_stream_state(KC_GET_STREAM(space));
      req.add_callback([space, rv, pckd_rv]() {
        RecvPacker::unpack_into(space, rv, pckd_rv.view_);
        space.fence("fence `pckd_rv` unpacking after " KC_GPU_NCCL_BACKEND_NAME " call");
      });
    }
  } else {
    auto pckd_sv = SendPacker::pack(space, "pckd_sv", sv);
    if (rank != root and is_contiguous(rv)) {
      ncclReduce(data_handle(pckd_sv.view_), data_handle(rv), pckd_sv.count_, pckd_sv.datatype_, op, root, comm,
                 KC_GET_STREAM(space));
      req.capture_stream_state(KC_GET_STREAM(space));
    } else {
      auto pckd_rv = RecvPacker::allocate_packed_for(space, "pckd_rv", rv);
      ncclReduce(data_handle(pckd_sv.view_), data_handle(pckd_rv.view_), pckd_sv.count_, pckd_sv.datatype_, op, root,
                 comm, KC_GET_STREAM(space));
      req.capture_stream_state(KC_GET_STREAM(space));
      req.add_callback([space, rv, pckd_rv]() {
        RecvPacker::unpack_into(space, rv, pckd_rv.view_);
        space.fence("fence `pckd_rv` unpacking after " KC_GPU_NCCL_BACKEND_NAME " call");
      });
    }
    req.extend_view_lifetime(pckd_sv.view_);
  }
  req.extend_view_lifetime(sv);
  req.extend_view_lifetime(rv);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace gpu
namespace Impl {

template <KokkosView SendView, KokkosView RecvView, ReductionOperator RedOp>
struct Reduce<SendView, RecvView, RedOp, KC_GPU_EXEC_SPACE, GpuCommSpace> {
  static auto execute(Handle<KC_GPU_EXEC_SPACE, GpuCommSpace>& h, const SendView sv, RecvView rv, int root)
      -> Request<GpuCommSpace> {
    return gpu::reduce(h.space(), sv, rv, reduction_op<GpuCommSpace, RedOp>(), root, h.rank(), h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm::Experimental
