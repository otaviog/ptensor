set(_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/ptensor/op)
add_library(ptensor_op)

find_package(kissfft CONFIG REQUIRED)

##
# Public Headers
set(PUBLIC_HEADERS
  ${_INCLUDE_DIR}/blur.hpp
  ${_INCLUDE_DIR}/elemwise.hpp
  ${_INCLUDE_DIR}/resize.hpp
  ${_INCLUDE_DIR}/image.hpp
  ${_INCLUDE_DIR}/laplacian_pyramid.hpp
  ${_INCLUDE_DIR}/fft.hpp
  ${_INCLUDE_DIR}/tensor_scalar.hpp
  ${_INCLUDE_DIR}/stack.hpp
  ${_INCLUDE_DIR}/image_rgb_to_gray.hpp
  ${_INCLUDE_DIR}/wave.hpp
  ${_INCLUDE_DIR}/window_function.hpp
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
    src/blur.cpp
    src/elemwise.cpp
    src/tensor_scalar.cpp
    src/resize.cpp
    src/image.cpp
    src/laplacian_pyramid.cpp
    src/fft.cpp
    src/stack.cpp
    src/image_rgb_to_gray.cpp
    src/wave.cpp
    src/window_function.cpp
)

target_include_directories(ptensor_op
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE
    ${_INCLUDE_DIR})

target_link_libraries(ptensor_op PUBLIC ptensor PRIVATE kissfft::kissfft-float)

# Explicitly set what dependencies are exported for consumers
# kissfft is a private implementation detail and should not be exposed
set_property(TARGET ptensor_op PROPERTY INTERFACE_LINK_LIBRARIES ptensor)

ptensor_target_options(ptensor_op Op)

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

#### Installation
# Create alias for use in build tree and install
add_library(p10::op ALIAS ptensor_op)
install(TARGETS ptensor_op
    EXPORT ptensorTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    FILE_SET public_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
