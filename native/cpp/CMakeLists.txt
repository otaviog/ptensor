add_subdirectory(core)

set(WITH_IO OFF CACHE BOOL "Build Input/Output library")
set(WITH_OP OFF CACHE BOOL "Build Operations library")

add_subdirectory(testing)
if (BUILD_TESTS)
  set(WITH_IO TRUE)
endif()

if (WITH_IO)
  add_subdirectory(io)
endif (WITH_IO)

if (WITH_OP)
  add_subdirectory(op)
endif (WITH_OP)