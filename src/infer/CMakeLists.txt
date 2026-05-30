set(_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/ptensor/infer)
set(onnxruntime_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/onnxruntime-osx-arm64-1.24.4/lib/cmake/onnxruntime)
find_package(onnxruntime CONFIG REQUIRED)
find_package(CURL REQUIRED)

add_library(ptensor_infer STATIC)

set(PUBLIC_HEADERS
  ${_INCLUDE_DIR}/infer.hpp
  ${_INCLUDE_DIR}/infer_config.hpp
)

target_sources(ptensor_infer
  PUBLIC
    FILE_SET public_headers TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
    FILES ${PUBLIC_HEADERS}
  PRIVATE
    src/infer.cpp
    src/model_downloader.hpp src/model_downloader.cpp
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
    ptensor_log
    CURL::libcurl)

install(TARGETS ptensor_infer
    ARCHIVE EXCLUDE_FROM_ALL
    FILE_SET public_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES $<TARGET_FILE:onnxruntime::onnxruntime>
    DESTINATION ${PTENSOR_INSTALL_LIB_DIR}
)

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()
