set(_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/ptensor/recog)
add_library(ptensor_recog STATIC)

set(PUBLIC_HEADERS
  ${_INCLUDE_DIR}/face_detection.hpp
  ${_INCLUDE_DIR}/face_detector.hpp
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
    src/face_detection.cpp
    src/face_detector.cpp
    # Common algorithms
    src/nms.hpp src/nms.cpp
    # Ssd anchors
    src/ssd_anchors/ssd_anchor_rect.hpp
    src/ssd_anchors/ssd_anchor_cache.hpp src/ssd_anchors/ssd_anchor_cache.cpp
    src/ssd_anchors/ssd_anchor_decoder.hpp src/ssd_anchors/ssd_anchor_decoder.cpp
    # Blaze face
    src/blaze_face/blaze_face.hpp src/blaze_face/blaze_face.cpp
    src/blaze_face/bf_preprocess.hpp src/blaze_face/bf_preprocess.cpp
    src/blaze_face/bf_postprocess.hpp src/blaze_face/bf_postprocess.cpp)

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

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

if (BUILD_SAMPLES)
  add_subdirectory(apps/face_detect)
endif()

