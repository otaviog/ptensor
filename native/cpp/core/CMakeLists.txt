set(_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/ptensor)
add_library(ptensor)

##
# Public Headers
set(PUBLIC_HEADERS
  ${_INCLUDE_DIR}/p10_error.hpp
  ${_INCLUDE_DIR}/p10_result.hpp
  ${_INCLUDE_DIR}/axis.hpp
  ${_INCLUDE_DIR}/device.hpp
  ${_INCLUDE_DIR}/dtype.hpp
  ${_INCLUDE_DIR}/shape.hpp
  ${_INCLUDE_DIR}/stride.hpp
  ${_INCLUDE_DIR}/span2d.hpp
  ${_INCLUDE_DIR}/span3d.hpp
  ${_INCLUDE_DIR}/tensor.hpp
  ${_INCLUDE_DIR}/tensor_options.hpp
  ${_INCLUDE_DIR}/tensor_print.hpp
  ${_INCLUDE_DIR}/detail/blob.hpp
  ${_INCLUDE_DIR}/detail/string.hpp
  ${_INCLUDE_DIR}/detail/panic.hpp
  ${_INCLUDE_DIR}/detail/tensor_extents.hpp
  ${_INCLUDE_DIR}/detail/complex_traits.hpp
  )

target_sources(ptensor
    PUBLIC
    FILE_SET public_headers TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
    FILES
    ${PUBLIC_HEADERS})
set_source_files_properties(${PUBLIC_HEADERS} PROPERTIES VS_FOLDER "Public Headers")

##
# Source to public headers
target_sources(ptensor
    PRIVATE
    src/p10_error.cpp
    src/stride.cpp
    src/tensor.cpp
    src/tensor.transpose.cpp
    src/tensor_print.cpp
    src/detail/blob.cpp
    src/dtype.cpp
)

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

target_link_libraries(ptensor PRIVATE $<BUILD_INTERFACE:ptensor_simd_>)
# Add C API headers that are referenced by C++ headers
install(FILES
    ${CMAKE_SOURCE_DIR}/native/c/include/ptensor/config.h
    ${CMAKE_SOURCE_DIR}/native/c/include/ptensor/ptensor_error.h
    ${CMAKE_SOURCE_DIR}/native/c/include/ptensor/ptensor_dtype.h
    ${CMAKE_SOURCE_DIR}/native/c/include/ptensor/ptensor_device.h
    ${CMAKE_SOURCE_DIR}/native/c/include/ptensor/ptensor_int_types.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ptensor)

ptensor_target_options(ptensor "Core")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

#### Installation
# Create alias for use in build tree and install
add_library(p10::core ALIAS ptensor)
install(TARGETS ptensor
    EXPORT ptensorTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    FILE_SET public_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
