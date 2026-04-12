set(_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/ptensor/recog)
add_library(ptensor_recog STATIC)

set(PUBLIC_HEADERS
  ${_INCLUDE_DIR}/face_detection.hpp
  ${_INCLUDE_DIR}/geom/point2.hpp
  ${_INCLUDE_DIR}/geom/rect2.hpp
)

target_sources(ptensor_recog
  PUBLIC
  FILE_SET public_headers TYPE HEADERS
  BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
  FILES ${PUBLIC_HEADERS}
  PRIVATE
  src/face_detection.cpp
  src/anchors.hpp
  src/blaze_face/blaze_face.hpp src/blaze_face/blaze_face.cpp
  src/blaze_face/bf_preprocess.hpp src/blaze_face/bf_preprocess.cpp
  src/blaze_face/bf_postprocess.hpp src/blaze_face/bf_postprocess.cpp
  src/blaze_face/bf_anchors.hpp src/blaze_face/bf_anchors.cpp
  src/nms.hpp src/nms.cpp)

target_include_directories(
  ptensor_recog
  PRIVATE
  ${_INCLUDE_DIR})

target_link_libraries(ptensor_recog
  PUBLIC
  ptensor
  PRIVATE
  ptensor_infer
  ptensor_op
    # ng-log::ng-log
  )

if(BUILD_TESTS)
  #add_subdirectory(tests)
endif()

