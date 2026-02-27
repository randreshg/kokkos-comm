target_sources(
  KokkosComm
  INTERFACE
    # Public GPU headers
    FILE_SET HEADERS
      BASE_DIRS ${PROJECT_SOURCE_DIR}/src
      FILES
        # Runtime abstraction
        gpu_runtime.hpp
        # Objects
        handle.hpp
        request.hpp
        # Point-to-point
        send.hpp
        recv.hpp
        # Collectives
        broadcast.hpp
        allgather.hpp
        alltoall.hpp
        allreduce.hpp
        reduce.hpp
    # Implementation detail GPU headers
    FILE_SET HEADERS
      BASE_DIRS ${PROJECT_SOURCE_DIR}/src
      FILES impl/pack_traits.hpp impl/packer.hpp impl/error_handling.hpp
)
