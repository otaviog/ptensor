add_subdirectory(simd)
add_subdirectory(log)
add_subdirectory(core)

set(WITH_IO OFF CACHE BOOL "Build Input/Output library")
set(WITH_OP OFF CACHE BOOL "Build Operations library")
set(WITH_MEDIA OFF CACHE BOOL "Build Media library (requires FFMPEG)")
set(WITH_INFER OFF CACHE BOOL "Build Infer library (requires OnnxRuntime)")
set(WITH_RECOG OFF CACHE BOOL "Build Recog library (requires ptensor_infer)")
set(WITH_VIZ OFF CACHE BOOL "Build Viz library (ImGui-based, requires ptensor_media)")

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

if (WITH_INFER)
  add_subdirectory(infer)
endif (WITH_INFER)

if (WITH_RECOG)
  add_subdirectory(recog)
endif (WITH_RECOG)

if (WITH_VIZ)
  if (NOT WITH_MEDIA)
    message(FATAL_ERROR "WITH_VIZ requires WITH_MEDIA")
  endif()
  add_subdirectory(viz)
endif (WITH_VIZ)

# The C API is built last so optional module bindings can be included.
add_subdirectory(c)
