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
    src/time/time.cpp
    src/io/media_device.cpp
    src/io/media_capture.cpp
    # Common writer
    src/io/media_writer.impl.hpp
    src/io/media_writer.cpp
    # FFMPEG utils
    src/io/ffmpeg/ffmpeg_memory.hpp
    # FFMPEG data and processing
    src/io/ffmpeg/ffmpeg_audio_fifo.hpp src/io/ffmpeg/ffmpeg_audio_fifo.cpp
    src/io/ffmpeg/ffmpeg_sws.hpp src/io/ffmpeg/ffmpeg_sws.cpp
    src/io/ffmpeg/ffmpeg_swr.hpp src/io/ffmpeg/ffmpeg_swr.cpp
    # FFMPEG capture
    src/io/ffmpeg/ffmpeg_media_capture_engine.hpp src/io/ffmpeg/ffmpeg_media_capture_engine.cpp
    src/io/ffmpeg/ffmpeg_file_media_capture.hpp src/io/ffmpeg/ffmpeg_file_media_capture.cpp
    src/io/ffmpeg/ffmpeg_device_media_capture.hpp src/io/ffmpeg/ffmpeg_device_media_capture.cpp
    src/io/ffmpeg/ffmpeg_device_enum.cpp
    src/io/ffmpeg/ffmpeg_video_decoder.hpp src/io/ffmpeg/ffmpeg_video_decoder.cpp
    # FFMPEG writer
    src/io/ffmpeg/ffmpeg_media_writer.hpp src/io/ffmpeg/ffmpeg_media_writer.cpp
    src/io/ffmpeg/ffmpeg_video_encoder.hpp src/io/ffmpeg/ffmpeg_video_encoder.cpp
    src/io/ffmpeg/ffmpeg_audio_encoder.hpp src/io/ffmpeg/ffmpeg_audio_encoder.cpp
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

