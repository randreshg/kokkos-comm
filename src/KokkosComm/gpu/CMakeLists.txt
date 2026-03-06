target_sources(
  KokkosComm
  INTERFACE
    # Public GPU headers
    FILE_SET HEADERS
      BASE_DIRS ${PROJECT_SOURCE_DIR}/src
      FILES
        # GPU communication space (trait-based abstraction, like mpi_space.hpp)
        gpu_space.hpp
        backends/nccl_device.hpp
        backends/rccl_device.hpp
        # Objects
        handle.hpp
        request.hpp
        # Point-to-point
        send.hpp
        recv.hpp
        sendrecv.hpp
        # Collectives
        broadcast.hpp
        allgather.hpp
        alltoall.hpp
        allreduce.hpp
        reduce.hpp
    # Implementation detail GPU headers
    FILE_SET HEADERS
      BASE_DIRS ${PROJECT_SOURCE_DIR}/src
      FILES impl/pack_traits.hpp impl/packer.hpp impl/error_handling.hpp impl/group_semantics.hpp
)

# Link GPU communication library (NCCL or RCCL)
if(KokkosComm_ENABLE_NCCL)
  target_link_libraries(KokkosComm INTERFACE NCCL::NCCL)
endif()
if(KokkosComm_ENABLE_RCCL)
  target_link_libraries(KokkosComm INTERFACE rccl)
endif()
