add_library(unit_tests_op OBJECT
    test_elemwise.cpp
    test_image.cpp
    test_laplacian.cpp
)
ptensor_target_options(unit_tests_op Op)
target_link_libraries(unit_tests_op
    PUBLIC ptensor ptensor_op PRIVATE Catch2::Catch2 ptensor_testing ptensor_io)
