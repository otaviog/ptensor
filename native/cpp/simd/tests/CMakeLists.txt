add_library(unit_tests_simd OBJECT test_bitwise.cpp)
target_link_libraries(unit_tests_simd
    PUBLIC ptensor_simd_
    PRIVATE Catch2::Catch2)
ptensor_target_options(unit_tests_simd "Tests")