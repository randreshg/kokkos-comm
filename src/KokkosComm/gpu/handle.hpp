// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core.hpp>

#include <KokkosComm/fwd.hpp>
#include <KokkosComm/gpu/gpu_space.hpp>

namespace KokkosComm {

template <>
class Handle<Experimental::GpuCommSpace::device_traits::exec_space, Experimental::GpuCommSpace> {
 public:
  using execution_space     = typename Experimental::GpuCommSpace::device_traits::exec_space;
  using communication_space = Experimental::GpuCommSpace;
  using handle_type         = communication_space::handle_type;
  using datatype_type       = communication_space::datatype_type;
  using reduction_op_type   = communication_space::reduction_op_type;
  using rank_type           = communication_space::rank_type;

  explicit Handle(const execution_space& space, handle_type comm) : space_(space), comm_(comm) {}
  explicit Handle(handle_type comm) : Handle(execution_space{}, comm) {}

  auto comm() -> handle_type& { return comm_; }
  auto comm() const -> const handle_type& { return comm_; }
  auto space() -> execution_space& { return space_; }
  auto space() const -> const execution_space& { return space_; }

  auto size() -> rank_type {
    rank_type ret;
    ncclCommCount(comm_, &ret);
    return ret;
  }

  auto rank() -> rank_type {
    rank_type ret;
    ncclCommUserRank(comm_, &ret);
    return ret;
  }

 private:
  execution_space space_;
  handle_type comm_;
};

}  // namespace KokkosComm
