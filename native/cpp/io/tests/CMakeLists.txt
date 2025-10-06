add_library(unit_tests_io OBJECT
    test_numpy.cpp
    test_image.cpp
)
ptensor_target_options(unit_tests_io Tests)
target_link_libraries(unit_tests_io
    PUBLIC ptensor ptensor_io PRIVATE Catch2::Catch2 ptensor_testing)
