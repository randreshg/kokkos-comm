// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <string>

#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/impl/contiguous.hpp>

#include <KokkosComm/gpu/gpu_runtime.hpp>

namespace KokkosComm::Experimental::gpu::Impl::Packer {

template <KokkosView View>
struct PackedGpuView {
  View view_;
  ncclDataType_t datatype_;
  int count_;

  PackedGpuView(const View& view, ncclDataType_t datatype, int count)
      : view_(view), datatype_(datatype), count_(count) {}
};

template <KokkosView V>
struct DeepCopy {
  using PackedV = KokkosComm::Impl::contiguous_view_t<V>;
  using T       = typename PackedV::non_const_value_type;

  template <KokkosExecutionSpace E>
  static auto allocate_packed_for(const E& exec, const std::string& label, const V& view)
      -> PackedGpuView<PackedV> {
    auto packed = KokkosComm::Impl::allocate_contiguous_for(exec, label, view);
    return PackedGpuView<PackedV>(packed, datatype<GpuCommSpace, T>(), span(packed));
  }

  template <KokkosExecutionSpace E>
  static auto pack(const E& exec, const std::string& label, const V& view) -> PackedGpuView<PackedV> {
    auto packed = allocate_packed_for(exec, label, view);
    Kokkos::deep_copy(exec, packed.view_, view);
    return packed;
  }

  template <KokkosExecutionSpace E>
  static auto unpack_into(const E& exec, const V& dst, const PackedV& src) -> void {
    Kokkos::deep_copy(exec, dst, src);
  }
};

}  // namespace KokkosComm::Experimental::gpu::Impl::Packer
