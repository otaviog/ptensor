set(_INCLUDE_DIR include/ptensor/testing)
add_library(ptensor_testing INTERFACE)

target_sources(ptensor_testing INTERFACE FILE_SET public_headers
    TYPE HEADERS FILES
    ${_INCLUDE_DIR}/catch2_assertions.hpp)

target_include_directories(ptensor_testing INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(ptensor_testing INTERFACE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>)

target_link_libraries(ptensor_testing INTERFACE ptensor)

# Create alias for use in build tree and install
add_library(p10::testing ALIAS ptensor_testing)

# Installation and export configuration
include(GNUInstallDirs)

# Install the library and headers
install(TARGETS ptensor_testing
    EXPORT ptensorTargets
    FILE_SET public_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if (BUILD_TESTS)
    add_subdirectory(tests)
endif()
