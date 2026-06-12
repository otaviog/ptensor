add_library(unit_tests_simd OBJECT test_bitwise.cpp test_tile.cpp)
target_link_libraries(unit_tests_simd
    PUBLIC ptensor_simd_ ptensor ptensor_testing
    PRIVATE Catch2::Catch2)
ptensor_target_options(unit_tests_simd "Core")
