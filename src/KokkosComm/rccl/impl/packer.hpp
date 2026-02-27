// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once
#include <KokkosComm/gpu/impl/packer.hpp>

// Backward-compatible type aliases
namespace KokkosComm::Experimental::rccl::Impl::Packer {

template <KokkosView View>
using PackedRcclView = KokkosComm::Experimental::gpu::Impl::Packer::PackedGpuView<View>;

template <KokkosView V>
using DeepCopy = KokkosComm::Experimental::gpu::Impl::Packer::DeepCopy<V>;

}  // namespace KokkosComm::Experimental::rccl::Impl::Packer
