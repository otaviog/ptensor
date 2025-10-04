add_library(unit_tests_core OBJECT test_tensor.cpp test_ptensor_error.cpp test_shape.cpp test_dtype.cpp)
target_link_libraries(unit_tests_core 
    PUBLIC ptensor PRIVATE Catch2::Catch2)
