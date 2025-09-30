set(_INCLUDE_DIR include/ptensor/testing)
add_library(ptensor_testing INTERFACE)

target_sources(ptensor_testing INTERFACE FILE_SET public_headers
    TYPE HEADERS FILES
    ${_INCLUDE_DIR}/catch2_assertions.hpp)
target_include_directories(ptensor_testing INTERFACE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
target_compile_options(ptensor_testing INTERFACE
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>)
target_link_libraries(ptensor_testing INTERFACE ptensor)

if (BUILD_TESTS)
    add_subdirectory(tests)
endif()
