add_subdirectory(core)

set(WITH_IO OFF CACHE BOOL "Build IO library")

if (BUILD_TESTS)
  set(WITH_IO TRUE)
  add_subdirectory(testing)
endif()

if (WITH_IO)
  add_subdirectory(io)
endif (WITH_IO)
