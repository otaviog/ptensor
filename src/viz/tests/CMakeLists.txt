add_library(unit_tests_viz OBJECT
    test_timer.cpp
)
target_link_libraries(unit_tests_viz
    PUBLIC ptensor ptensor_viz
    PRIVATE Catch2::Catch2 ptensor_testing
)
ptensor_target_options(unit_tests_viz Viz)
