cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS On)

# Create compile_commands.json
include("./cmake/export_compile_commands.cmake")
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
endif()

add_subdirectory(native)
