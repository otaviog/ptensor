cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS On)

# Configure installation directories
include(GNUInstallDirs)

# Create compile_commands.json
include("./cmake/export_compile_commands.cmake")
include("./cmake/ptensor_target_options.cmake")

set(BUILD_TESTS OFF CACHE BOOL "Build tests")
if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0 # or a later release
    )
    FetchContent_MakeAvailable(Catch2)
    include(CTest)
    include(Catch)
    enable_testing()
    set_target_properties(Catch2 PROPERTIES
        FOLDER Libraries)
    set_target_properties(Catch2WithMain PROPERTIES
        FOLDER Libraries)

    find_package(Threads REQUIRED)
    find_package(benchmark CONFIG REQUIRED)
endif()
set(BUILD_SAMPLES OFF CACHE BOOL "Build sample applications")

set(PTENSOR_INSTALL_LIB_DIR $<IF:$<CONFIG:Debug>,debug/lib,lib>)
set(PTENSOR_INSTALL_BIN_DIR $<IF:$<CONFIG:Debug>,debug/bin,bin>)

add_subdirectory(native)

if (EMSCRIPTEN)
    add_subdirectory(wasm/cpp)
endif()

# Create and install the config files
# The export will generate ptensorTargets.cmake and configuration-specific files
# like ptensorTargets-debug.cmake and ptensorTargets-release.cmake
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
