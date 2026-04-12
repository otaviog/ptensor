set(_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/ptensor/infer)
set(onnxruntime_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/onnxruntime-osx-arm64-1.24.4/lib/cmake/onnxruntime)
find_package(onnxruntime CONFIG REQUIRED)

find_package(ng-log CONFIG REQUIRED)

add_library(ptensor_infer STATIC)

set(PUBLIC_HEADERS
  ${_INCLUDE_DIR}/infer.hpp
)

target_sources(ptensor_infer
  PUBLIC
    FILE_SET public_headers TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
    FILES ${PUBLIC_HEADERS}
  PRIVATE
    src/infer.cpp
    src/ort_conversions.hpp
    src/ort_infer.hpp src/ort_infer.cpp)

target_include_directories(
  ptensor_infer
  PRIVATE
  ${_INCLUDE_DIR})

target_link_libraries(ptensor_infer
  PUBLIC
    ptensor
  PRIVATE
    onnxruntime::onnxruntime
    ng-log::ng-log)

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()
