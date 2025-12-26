find_package(FFMPEG REQUIRED)

set(_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/ptensor/media)
add_library(ptensor_media STATIC)

set(PUBLIC_HEADERS
  ${_INCLUDE_DIR}/video_frame.hpp
  ${_INCLUDE_DIR}/audio_frame.hpp
  ${_INCLUDE_DIR}/time/time.hpp
  ${_INCLUDE_DIR}/time/rational.hpp
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
  ${FFMPEG_INCLUDE_DIRS}
)
target_link_directories(ptensor_media PRIVATE ${FFMPEG_LIBRARY_DIRS})
target_link_libraries(ptensor_media PUBLIC ptensor PRIVATE ${FFMPEG_LIBRARIES})

ptensor_target_options(ptensor_media "MEDIA")

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
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  FILE_SET public_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
