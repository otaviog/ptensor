find_package(FFMPEG REQUIRED)

# clang's driver already links libc++ implicitly; FFMPEG_LIBRARIES also lists
# bare `c++`, which produces a duplicate-library link warning on macOS.
set(_FFMPEG_LIBRARIES ${FFMPEG_LIBRARIES})
list(TRANSFORM _FFMPEG_LIBRARIES REPLACE "^c\\+\\+$" "")

set(_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/ptensor/media)
add_library(ptensor_media STATIC)

set(PUBLIC_HEADERS
  ${_INCLUDE_DIR}/video_frame.hpp
  ${_INCLUDE_DIR}/audio_frame.hpp
  ${_INCLUDE_DIR}/time/time.hpp
  ${_INCLUDE_DIR}/time/rational.hpp
  ${_INCLUDE_DIR}/io/media_capture.hpp
  ${_INCLUDE_DIR}/io/media_device.hpp
  ${_INCLUDE_DIR}/io/media_writer.hpp
  ${_INCLUDE_DIR}/media_parameters.hpp
  ${_INCLUDE_DIR}/audio_parameters.hpp
  ${_INCLUDE_DIR}/video_parameters.hpp
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
    # Common writer
    io/media_writer.impl.hpp
    io/media_writer.cpp
    # FFMPEG utils
    io/ffmpeg/ffmpeg_memory.hpp
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
    # FFMPEG writer
    io/ffmpeg/ffmpeg_media_writer.hpp io/ffmpeg/ffmpeg_media_writer.cpp
    io/ffmpeg/ffmpeg_video_encoder.hpp io/ffmpeg/ffmpeg_video_encoder.cpp
    io/ffmpeg/ffmpeg_audio_encoder.hpp io/ffmpeg/ffmpeg_audio_encoder.cpp
)

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
  add_subdirectory(samples)
endif()

# Install the library and headers
install(TARGETS ptensor_media
    ARCHIVE EXCLUDE_FROM_ALL
    FILE_SET public_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

