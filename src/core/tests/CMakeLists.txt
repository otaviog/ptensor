add_library(unit_tests_core OBJECT test_tensor.cpp test_ptensor_error.cpp test_shape.cpp test_stride.cpp test_dtype.cpp test_tensor_print.cpp)
ptensor_target_options(unit_tests_core Core)
target_link_libraries(unit_tests_core 
    PUBLIC ptensor PRIVATE Catch2::Catch2 ptensor_testing)
