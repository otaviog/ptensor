include(FetchContent)

set(_ORT_VERSION "1.24.4")

if(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        set(_ORT_PLATFORM "osx-arm64")
        set(_ORT_ARCHIVE_EXT "tgz")
    else()
        set(_ORT_PLATFORM "osx-x86_64")
        set(_ORT_ARCHIVE_EXT "tgz")
    endif()
elseif(WIN32)
    set(_ORT_PLATFORM "win-x64")
    set(_ORT_ARCHIVE_EXT "zip")
else()
    set(_ORT_PLATFORM "linux-x64")
    set(_ORT_ARCHIVE_EXT "tgz")
endif()

set(_ORT_DIR_NAME "onnxruntime-${_ORT_PLATFORM}-${_ORT_VERSION}")
set(_ORT_LOCAL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/lib/${_ORT_DIR_NAME}")

if(NOT EXISTS "${_ORT_LOCAL_PATH}")
    set(_ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${_ORT_VERSION}/${_ORT_DIR_NAME}.${_ORT_ARCHIVE_EXT}")
    message(STATUS "Downloading ONNX Runtime ${_ORT_VERSION} for ${_ORT_PLATFORM}...")
    FetchContent_Declare(
        onnxruntime_prebuilt
        URL "${_ORT_URL}"
        SOURCE_DIR "${_ORT_LOCAL_PATH}"
    )
    FetchContent_MakeAvailable(onnxruntime_prebuilt)
    # The released cmake configs have wrong paths on Linux; patch them.
    # - lib64/ -> lib/ (library location)
    # - include/onnxruntime -> include (headers sit directly in include/)
    set(_ORT_TARGETS_FILE "${_ORT_LOCAL_PATH}/lib/cmake/onnxruntime/onnxruntimeTargets-release.cmake")
    if(EXISTS "${_ORT_TARGETS_FILE}")
        file(READ "${_ORT_TARGETS_FILE}" _ORT_CONTENT)
        string(REPLACE "/lib64/" "/lib/" _ORT_CONTENT "${_ORT_CONTENT}")
        file(WRITE "${_ORT_TARGETS_FILE}" "${_ORT_CONTENT}")
    endif()
    set(_ORT_MAIN_FILE "${_ORT_LOCAL_PATH}/lib/cmake/onnxruntime/onnxruntimeTargets.cmake")
    if(EXISTS "${_ORT_MAIN_FILE}")
        file(READ "${_ORT_MAIN_FILE}" _ORT_CONTENT)
        string(REPLACE "/include/onnxruntime" "/include" _ORT_CONTENT "${_ORT_CONTENT}")
        file(WRITE "${_ORT_MAIN_FILE}" "${_ORT_CONTENT}")
    endif()
endif()

set(onnxruntime_DIR "${_ORT_LOCAL_PATH}/lib/cmake/onnxruntime")
find_package(onnxruntime CONFIG REQUIRED)
find_package(CURL REQUIRED)

set(_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/ptensor/infer)

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
