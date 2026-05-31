set(_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/ptensor/infer)

# ONNX Runtime — bypass cmake config (vcpkg wrapping causes path resolution
# issues with prebuilt tarballs); declare the imported target directly.
set(_ORT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/lib/onnxruntime)
if(NOT EXISTS ${_ORT_ROOT}/include)
  message(STATUS "ONNX Runtime not found — running scripts/setup.ts")
  execute_process(
    COMMAND ${CMAKE_SOURCE_DIR}/scripts/run-bun.sh
            ${CMAKE_SOURCE_DIR}/scripts/setup.ts
    RESULT_VARIABLE _ORT_SETUP_RESULT
    COMMAND_ECHO STDOUT)
  if(NOT _ORT_SETUP_RESULT EQUAL 0)
    message(FATAL_ERROR "scripts/setup.ts failed. Run manually: bun scripts/setup.ts")
  endif()
endif()

find_library(_ORT_LIB
  NAMES onnxruntime
  PATHS ${_ORT_ROOT}/lib ${_ORT_ROOT}/lib64
  NO_DEFAULT_PATH REQUIRED)

add_library(onnxruntime::onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
  IMPORTED_LOCATION             "${_ORT_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${_ORT_ROOT}/include"
)
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
    infer.cpp
    model_downloader.hpp model_downloader.cpp
    ort_conversions.hpp
    ort_infer.hpp ort_infer.cpp)

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
