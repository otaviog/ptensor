add_subdirectory(core)

set(WITH_IO OFF CACHE BOOL "Build Input/Output library")
set(WITH_OP OFF CACHE BOOL "Build Operations library")
set(WITH_MEDIA OFF CACHE BOOL "Build Media library (requires FFMPEG)")

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

if (WITH_MEDIA)
  add_subdirectory(media)
endif (WITH_MEDIA)