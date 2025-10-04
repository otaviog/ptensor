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


# Installation and export configuration
include(GNUInstallDirs)

# Install the library and headers
install(TARGETS ptensor
    EXPORT ptensorTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    FILE_SET public_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Create and install the config files
install(EXPORT ptensorTargets
    FILE ptensorTargets.cmake
    NAMESPACE ptensor::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ptensor
)

# Create the config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ptensorConfigVersion.cmake
    VERSION 1.0.0
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/ptensorConfig.cmake.in
    ptensorConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ptensor
)

# Install the config files
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ptensorConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ptensorConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ptensor
)

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()
