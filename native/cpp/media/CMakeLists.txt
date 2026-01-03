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
    src/io/media_capture.cpp
    src/io/ffmpeg/ffmpeg_file_media_capture.hpp src/io/ffmpeg/ffmpeg_file_media_capture.cpp
    src/io/ffmpeg/ffmpeg_sws.hpp src/io/ffmpeg/ffmpeg_sws.cpp
    src/io/ffmpeg/ffmpeg_video_decoder.hpp src/io/ffmpeg/ffmpeg_video_decoder.cpp
)

target_include_directories(ptensor_media
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${_INCLUDE_DIR}
    $<BUILD_INTERFACE:${FFMPEG_INCLUDE_DIRS}>
)
# On Windows with SHARED libraries (DLL), consumers need access to FFmpeg libraries
# because symbols aren't automatically re-exported from DLLs
if(WIN32)
  set(_FFMPEG_LINK_SCOPE PUBLIC)
else()
  set(_FFMPEG_LINK_SCOPE PRIVATE)
endif()

target_link_libraries(ptensor_media
  PUBLIC ptensor
  ${_FFMPEG_LINK_SCOPE}
    ${FFMPEG_LIBRARIES}
    WebP::webp WebP::webpdecoder WebP::webpdemux WebP::libwebpmux
    PkgConfig::x265
)

ptensor_target_options(ptensor_media "Media")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(BUILD_SAMPLES)
  add_subdirectory(samples/video_app)
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
