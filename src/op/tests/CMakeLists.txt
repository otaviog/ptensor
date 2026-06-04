add_library(unit_tests_op OBJECT
    testing.hpp testing.cpp
    test_elemwise.cpp
    test_image.cpp
    test_crop.cpp
    test_laplacian_pyramid.cpp
    test_resize.cpp
    test_blur.cpp
    test_fft.cpp
    test_tensor_scalar.cpp
    test_stack.cpp
    test_image_rgb_to_gray.cpp
    test_statistics.cpp
)
ptensor_target_options(unit_tests_op Op)
target_link_libraries(unit_tests_op
    PUBLIC ptensor ptensor_op PRIVATE Catch2::Catch2 ptensor_testing ptensor_io)

add_subdirectory(benchmarks)
