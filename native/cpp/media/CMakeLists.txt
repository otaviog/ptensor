find_package(FFMPEG REQUIRED)
find_package(WebP CONFIG REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(x265 REQUIRED IMPORTED_TARGET x265)

set(_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/ptensor/media)
add_library(ptensor_media SHARED)

# On Windows, automatically export all symbols to generate the import library (.lib)
set_target_properties(ptensor_media PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

set(PUBLIC_HEADERS
  ${_INCLUDE_DIR}/video_frame.hpp
  ${_INCLUDE_DIR}/audio_frame.hpp
  ${_INCLUDE_DIR}/time/time.hpp
  ${_INCLUDE_DIR}/time/rational.hpp
  ${_INCLUDE_DIR}/io/media_capture.hpp
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
    src/io/ffmpeg/ffmpeg_file_media_capture.hpp src/io/ffmpeg/ffmpeg_file_media_capture.cpp
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
# Dependencies will be re-attached in ptensorConfig.cmake for consumers
target_link_libraries(ptensor_media
  PUBLIC ptensor
  PRIVATE
    ${FFMPEG_LIBRARIES}
    WebP::webp WebP::webpdecoder WebP::webpdemux WebP::libwebpmux
    PkgConfig::x265
)

ptensor_target_options(ptensor_media "Media")

if(BUILD_TESTS)
  add_subdirectory(tests)
  add_subdirectory(samples)
endif()

# Create alias for use in build tree and install
add_library(p10::media ALIAS ptensor_media)

# Install the library and headers
install(TARGETS ptensor_media
  EXPORT ptensorTargets
  LIBRARY DESTINATION ${PTENSOR_INSTALL_LIB_DIR}
  ARCHIVE DESTINATION ${PTENSOR_INSTALL_LIB_DIR}
  RUNTIME DESTINATION ${PTENSOR_INSTALL_BIN_DIR}
  FILE_SET public_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if (MSVC)
  install(FILES $<TARGET_PDB_FILE:ptensor_media> DESTINATION bin
          CONFIGURATIONS Debug)
endif()
