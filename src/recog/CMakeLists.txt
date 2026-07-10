set(_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/ptensor/recog)
add_library(ptensor_recog STATIC)

set(PUBLIC_HEADERS
  ${_INCLUDE_DIR}/face_detection.hpp
  ${_INCLUDE_DIR}/face_detector.hpp
  ${_INCLUDE_DIR}/ssd_anchor_parameters.hpp
  ${_INCLUDE_DIR}/geom/point2.hpp
  ${_INCLUDE_DIR}/geom/rect2.hpp
)

target_sources(ptensor_recog
  PUBLIC
  FILE_SET public_headers TYPE HEADERS
  BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
  FILES ${PUBLIC_HEADERS}
  PRIVATE
    # Main detection factory
    face_detection.cpp
    face_detector.cpp
    # Common algorithms
    nms.hpp nms.cpp
    # Ssd anchors
    ssd_anchors/ssd_anchor_rect.hpp
    ssd_anchors/ssd_anchor_cache.hpp ssd_anchors/ssd_anchor_cache.cpp
    ssd_anchors/ssd_anchor_decoder.hpp ssd_anchors/ssd_anchor_decoder.cpp
    # Blaze face
    blaze_face/blaze_face.hpp blaze_face/blaze_face.cpp
    blaze_face/bf_preprocess.hpp blaze_face/bf_preprocess.cpp
    blaze_face/bf_postprocess.hpp blaze_face/bf_postprocess.cpp)

target_include_directories(
  ptensor_recog
  PRIVATE
  ${_INCLUDE_DIR})

target_link_libraries(ptensor_recog
  PUBLIC
    ptensor
    ptensor_infer
  PRIVATE
    ptensor_op
    ptensor_log
  )

install(TARGETS ptensor_recog
    ARCHIVE EXCLUDE_FROM_ALL
    FILE_SET public_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

if (BUILD_APPS)
  add_subdirectory(apps/face_detect)
endif()

