find_package(FFMPEG REQUIRED)

# Build _FFMPEG_LIBRARIES from FFMPEG_LIBRARIES, skipping c++ and its paired
# optimized/debug keyword. Clang links libc++ implicitly, and a dangling
# keyword left behind by a simple REPLACE produces consecutive-specifier
# CMake warnings.
set(_FFMPEG_LIBRARIES "")
set(_pending_kw "")
foreach(_lib IN LISTS FFMPEG_LIBRARIES)
  if(_lib STREQUAL "optimized" OR _lib STREQUAL "debug")
    set(_pending_kw "${_lib}")
  elseif(_lib STREQUAL "c++")
    set(_pending_kw "")
  elseif(NOT _lib STREQUAL "")
    if(_pending_kw)
      list(APPEND _FFMPEG_LIBRARIES "${_pending_kw}" "${_lib}")
      set(_pending_kw "")
    else()
      list(APPEND _FFMPEG_LIBRARIES "${_lib}")
    endif()
  endif()
endforeach()
unset(_pending_kw)
unset(_lib)

set(_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/ptensor/media)
add_library(ptensor_media STATIC)

set(PUBLIC_HEADERS
  ${_INCLUDE_DIR}/video_frame.hpp
  ${_INCLUDE_DIR}/audio_frame.hpp
  ${_INCLUDE_DIR}/text.hpp
  ${_INCLUDE_DIR}/text_streams.hpp
  ${_INCLUDE_DIR}/time/time.hpp
  ${_INCLUDE_DIR}/time/rational.hpp
  ${_INCLUDE_DIR}/io/camera_control.hpp
  ${_INCLUDE_DIR}/io/media_capture.hpp
  ${_INCLUDE_DIR}/io/media_device.hpp
  ${_INCLUDE_DIR}/io/media_writer.hpp
  ${_INCLUDE_DIR}/media_parameters.hpp
  ${_INCLUDE_DIR}/audio_parameters.hpp
  ${_INCLUDE_DIR}/video_parameters.hpp
  ${_INCLUDE_DIR}/text_parameters.hpp
)

target_sources(ptensor_media
  PUBLIC
    FILE_SET public_headers TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
    FILES ${PUBLIC_HEADERS}
  PRIVATE
    # Common
    time/time.cpp
    io/media_device.cpp
    io/media_capture.cpp
    io/camera_controls.hpp io/camera_controls.cpp
    io/text_streams.cpp
    # Common writer
    io/media_writer.impl.hpp
    io/media_writer.cpp
    # FFMPEG utils
    io/ffmpeg/ffmpeg_memory.hpp
    io/ffmpeg/ffmpeg_init.hpp io/ffmpeg/ffmpeg_init.cpp
    # FFMPEG data and processing
    io/ffmpeg/ffmpeg_audio_fifo.hpp io/ffmpeg/ffmpeg_audio_fifo.cpp
    io/ffmpeg/ffmpeg_sws.hpp io/ffmpeg/ffmpeg_sws.cpp
    io/ffmpeg/ffmpeg_swr.hpp io/ffmpeg/ffmpeg_swr.cpp
    # FFMPEG capture
    io/ffmpeg/ffmpeg_media_capture_engine.hpp io/ffmpeg/ffmpeg_media_capture_engine.cpp
    io/ffmpeg/ffmpeg_file_media_capture.hpp io/ffmpeg/ffmpeg_file_media_capture.cpp
    io/ffmpeg/ffmpeg_device_media_capture.hpp io/ffmpeg/ffmpeg_device_media_capture.cpp
    io/ffmpeg/ffmpeg_device_enum.cpp
    io/ffmpeg/ffmpeg_video_decoder.hpp io/ffmpeg/ffmpeg_video_decoder.cpp
    io/ffmpeg/ffmpeg_text_decoder.hpp io/ffmpeg/ffmpeg_text_decoder.cpp
    # FFMPEG writer
    io/ffmpeg/ffmpeg_media_writer.hpp io/ffmpeg/ffmpeg_media_writer.cpp
    io/ffmpeg/ffmpeg_video_encoder.hpp io/ffmpeg/ffmpeg_video_encoder.cpp
    io/ffmpeg/ffmpeg_audio_encoder.hpp io/ffmpeg/ffmpeg_audio_encoder.cpp
    io/ffmpeg/ffmpeg_text_encoder.hpp io/ffmpeg/ffmpeg_text_encoder.cpp
)

# macOS camera controls go through AVFoundation (Objective-C++).
if(APPLE)
  target_sources(ptensor_media PRIVATE io/camera_controls_avf.mm)
  set_source_files_properties(io/camera_controls_avf.mm PROPERTIES
    COMPILE_FLAGS "-fno-objc-arc")
  target_link_libraries(ptensor_media PRIVATE
    "-framework AVFoundation"
    "-framework CoreMedia"
    "-framework Foundation")
endif()

target_include_directories(ptensor_media
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${_INCLUDE_DIR}
    $<BUILD_INTERFACE:${FFMPEG_INCLUDE_DIRS}>
)

# Link all external dependencies as PRIVATE to avoid exporting absolute paths
# Dependencies will be re-attached in ptensorConfig.cmake for consumers.
# FFMPEG_LIBRARIES already pulls in webp/webpmux/sharpyuv/x265/c++ via the
# ffmpeg vcpkg port's transitive deps, so they are not listed explicitly.
target_link_libraries(ptensor_media
  PUBLIC ptensor
  PRIVATE
    ptensor_log
    ${_FFMPEG_LIBRARIES}
)

ptensor_target_options(ptensor_media "Media")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(BUILD_APPS)
  add_subdirectory(apps)
endif()

# Install the library and headers
install(TARGETS ptensor_media
    ARCHIVE EXCLUDE_FROM_ALL
    FILE_SET public_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

