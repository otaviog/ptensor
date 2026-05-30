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

# Aggregate shared library — bundles all enabled modules into libptensor.so.
# Core (ptensor.a) is always included; optional modules are added based on
# the same WITH_* flags used to enable their subdirectories above.
include(${CMAKE_SOURCE_DIR}/cmake/ptensor_bundle.cmake)

add_library(ptensor_shared SHARED ptensor_bundle.cpp)
set_target_properties(ptensor_shared PROPERTIES OUTPUT_NAME ptensor)
if(WIN32)
    set_target_properties(ptensor_shared PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

target_link_whole_archive(ptensor_shared ptensor)

if(WITH_IO)
    target_link_whole_archive(ptensor_shared ptensor_io)
endif()
if(WITH_OP)
    target_link_whole_archive(ptensor_shared ptensor_op)
endif()
if(WITH_MEDIA)
    target_link_whole_archive(ptensor_shared ptensor_media)
endif()
if(WITH_INFER)
    target_link_whole_archive(ptensor_shared ptensor_infer)
endif()
if(WITH_RECOG)
    target_link_whole_archive(ptensor_shared ptensor_recog)
endif()
if(WITH_VIZ)
    target_link_whole_archive(ptensor_shared ptensor_viz)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(ptensor_shared PRIVATE -Wl,-Bsymbolic)
endif()

target_include_directories(ptensor_shared PUBLIC
    $<INSTALL_INTERFACE:include>
)

install(TARGETS ptensor_shared
    EXPORT ptensorTargets
    LIBRARY DESTINATION ${PTENSOR_INSTALL_LIB_DIR}
    RUNTIME DESTINATION ${PTENSOR_INSTALL_BIN_DIR}
)

# The C API is built last so optional module bindings can be included.
add_subdirectory(c)
