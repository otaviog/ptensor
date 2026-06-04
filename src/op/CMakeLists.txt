set(_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/ptensor/op)
add_library(ptensor_op STATIC)

find_path(POCKETFFT_INCLUDE_DIRS "pocketfft_hdronly.h" REQUIRED)

##
# Public Headers
set(PUBLIC_HEADERS
  ${_INCLUDE_DIR}/blur.hpp
  ${_INCLUDE_DIR}/crop.hpp
  ${_INCLUDE_DIR}/elemwise.hpp
  ${_INCLUDE_DIR}/resize.hpp
  ${_INCLUDE_DIR}/image_layout.hpp
  ${_INCLUDE_DIR}/laplacian_pyramid.hpp
  ${_INCLUDE_DIR}/fft.hpp
  ${_INCLUDE_DIR}/tensor_scalar.hpp
  ${_INCLUDE_DIR}/stack.hpp
  ${_INCLUDE_DIR}/image_rgb_to_gray.hpp
  ${_INCLUDE_DIR}/wave.hpp
  ${_INCLUDE_DIR}/window_function.hpp
  ${_INCLUDE_DIR}/statistics.hpp
  )

target_sources(ptensor_op
    PUBLIC
    FILE_SET public_headers TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
    FILES
    ${PUBLIC_HEADERS})
set_source_files_properties(${PUBLIC_HEADERS} PROPERTIES VS_FOLDER "Public Headers")

##
# Source to public headers
target_sources(ptensor_op
    PRIVATE
    accumulator_traits.hpp
    blur.cpp
    crop.cpp
    elemwise.cpp
    tensor_scalar.cpp
    resize.cpp
    image_layout.cpp
    laplacian_pyramid.cpp
    fft.cpp
    stack.cpp
    image_rgb_to_gray.cpp
    wave.cpp
    window_function.cpp
    statistics.cpp
)

target_include_directories(ptensor_op
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE
    ${POCKETFFT_INCLUDE_DIRS}
    ${_INCLUDE_DIR})

target_link_libraries(ptensor_op PUBLIC ptensor PRIVATE $<BUILD_INTERFACE:ptensor_simd_>)

ptensor_target_options(ptensor_op Op)

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

#### Installation
install(TARGETS ptensor_op
    ARCHIVE EXCLUDE_FROM_ALL
    FILE_SET public_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
