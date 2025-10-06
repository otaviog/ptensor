set(_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/ptensor/op)
add_library(ptensor_op)

##
# Public Headers
set(PUBLIC_HEADERS
  ${_INCLUDE_DIR}/blur.hpp
  ${_INCLUDE_DIR}/elemwise.hpp
  ${_INCLUDE_DIR}/fft.hpp
  ${_INCLUDE_DIR}/resize.hpp
  )

target_sources(ptensor_op
    PUBLIC 
    FILE_SET public_headers TYPE HEADERS
    FILES
    ${PUBLIC_HEADERS})
set_source_files_properties(${PUBLIC_HEADERS} PROPERTIES VS_FOLDER "Public Headers")

##
# Source to public headers
target_sources(ptensor_op
    PRIVATE
    src/blur.cpp
    src/elemwise.cpp
    src/fft.cpp
    src/resize.cpp
)

target_include_directories(ptensor_op
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE
    ${_INCLUDE_DIR})

target_compile_options(ptensor_op PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>)

if(NOT WIN32)
    target_compile_options(ptensor_op PRIVATE -fPIC)
endif()

#### Installation
# Create alias for use in build tree and install
add_library(p10::core ALIAS ptensor_op)
include(GNUInstallDirs)
install(TARGETS ptensor_op
    EXPORT ptensorTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    FILE_SET public_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Create and install the config files
install(EXPORT ptensorTargets
    FILE ptensorTargets.cmake
    NAMESPACE p10::
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
