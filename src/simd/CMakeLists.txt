set(_INCLUDE_DIR include/p10_internal/simd)

if(EMSCRIPTEN)
    set(_CPUID_IMPL cpuid.wasm.cpp)
elseif(WIN32)
    set(_CPUID_IMPL cpuid.windows.cpp)
elseif(APPLE)
    set(_CPUID_IMPL cpuid.macos.cpp)
else()
    set(_CPUID_IMPL cpuid.linux.cpp)
endif()

add_library(ptensor_simd_ STATIC)

target_sources(ptensor_simd_
    PUBLIC FILE_SET public_headers
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
        FILES
        ${_INCLUDE_DIR}/bitwise_math.hpp
        ${_INCLUDE_DIR}/compiler.hpp
        ${_INCLUDE_DIR}/cpuid.hpp
        ${_INCLUDE_DIR}/tile1d.hpp
        ${_INCLUDE_DIR}/tile2d.hpp
    PRIVATE
        ${_CPUID_IMPL}
)

target_include_directories(ptensor_simd_ PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

ptensor_target_options(ptensor_simd_ "Core")

# PUBLIC so the OpenMP flag reaches every TU that compiles the tile2d pragma,
# including consumers that link this lib PRIVATE (the flag is a build-time usage
# requirement, applied to their own compilation).
if(OpenMP_CXX_FOUND)
    target_link_libraries(ptensor_simd_ PUBLIC OpenMP::OpenMP_CXX)
endif()

if (BUILD_TESTS)
    add_subdirectory(tests)
endif()
