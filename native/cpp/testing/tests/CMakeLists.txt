add_library(unit_tests_testing OBJECT test_catch2_assertions.cpp)
target_link_libraries(unit_tests_testing
    PUBLIC ptensor_testing
    PRIVATE Catch2::Catch2)
ptensor_target_options(unit_tests_testing "Tests")