set(_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/ptensor)
add_library(ptensor)

##
# Public Headers
set(PUBLIC_HEADERS
  ${_INCLUDE_DIR}/ptensor_error.hpp
  ${_INCLUDE_DIR}/ptensor_result.hpp
  ${_INCLUDE_DIR}/device.hpp
  ${_INCLUDE_DIR}/tensor.hpp
  ${_INCLUDE_DIR}/detail/blob.hpp
  ${_INCLUDE_DIR}/detail/string.hpp
  ${_INCLUDE_DIR}/detail/panic.hpp
  )

target_sources(ptensor
    PUBLIC 
    FILE_SET public_headers TYPE HEADERS
    FILES
    ${PUBLIC_HEADERS})
set_source_files_properties(${PUBLIC_HEADERS} PROPERTIES VS_FOLDER "Public Headers")

##
# Source to public headers
target_sources(ptensor
    PRIVATE
    src/ptensor_error.cpp
    src/tensor.cpp
    src/detail/blob.cpp
    src/dtype.cpp
)

##
# Private source files
#target_sources(ptensor
#    PRIVATE
#)

if(WIN32)
    target_sources(ptensor
        PRIVATE src/detail/string.cpp)
endif(WIN32)

target_include_directories(ptensor
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/native/c/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE
    ${_INCLUDE_DIR})

target_compile_options(ptensor PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>)

if(NOT WIN32)
    target_compile_options(ptensor PRIVATE -fPIC)
endif()


if(BUILD_TESTS)
  add_subdirectory(tests)
endif()
