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
    PRIVATE
        ${_CPUID_IMPL}
)

target_include_directories(ptensor_simd_ PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

ptensor_target_options(ptensor_simd_ "Core")

if (BUILD_TESTS)
    add_subdirectory(tests)
endif()
