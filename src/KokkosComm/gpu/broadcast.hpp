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

template <KokkosView View>
auto broadcast(const KC_GPU_EXEC_SPACE& space, View& v, int root, ncclComm_t comm) -> Request<GpuCommSpace> {
  using T = typename View::non_const_value_type;
  static_assert(KC::rank<View>() <= 1,
                "KokkosComm::Experimental::gpu::broadcast: Views with rank higher than 1 are not supported");
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::gpu::broadcast");

  Request<GpuCommSpace> req;
  if (KC::is_contiguous(v)) {
    ncclBcast(KC::data_handle(v), KC::span(v), datatype<GpuCommSpace, T>(), root, comm, KC_GET_STREAM(space));
    req.capture_stream_state(KC_GET_STREAM(space));
  } else {
    Kokkos::abort("KokkosComm::Experimental::gpu::broadcast: unimplemented for non-contiguous views");
  }
  req.extend_view_lifetime(v);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace gpu
namespace Impl {

template <KokkosView View>
struct Broadcast<View, KC_GPU_EXEC_SPACE, GpuCommSpace> {
  static auto execute(Handle<KC_GPU_EXEC_SPACE, GpuCommSpace>& h, View v, int root) -> Request<GpuCommSpace> {
    return gpu::broadcast(h.space(), v, root, h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm::Experimental
