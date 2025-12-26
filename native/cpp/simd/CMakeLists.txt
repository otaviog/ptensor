set(_INCLUDE_DIR include/ptensor/simd)
add_library(ptensor_simd_ INTERFACE)

target_sources(ptensor_simd_ INTERFACE FILE_SET public_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
    FILES
    ${_INCLUDE_DIR}/bitwise_math.hpp
    ${_INCLUDE_DIR}/compiler.hpp
    ${_INCLUDE_DIR}/cpuid.hpp
    )

target_include_directories(ptensor_simd_ INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

set_target_properties(ptensor_simd_ PROPERTIES FOLDER "Core")

if (BUILD_TESTS)
    add_subdirectory(tests)
endif()
