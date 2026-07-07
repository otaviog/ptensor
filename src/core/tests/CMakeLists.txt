add_library(unit_tests_core OBJECT test_tensor.cpp test_ptensor_error.cpp test_shape.cpp test_stride.cpp test_dtype.cpp test_tensor_print.cpp)
ptensor_target_options(unit_tests_core Core)
target_link_libraries(unit_tests_core
    PUBLIC ptensor PRIVATE Catch2::Catch2 ptensor_testing)

# Optional map adapters: tests are built only when the third-party library is
# available. The headers themselves compile to nothing without it.
find_package(Eigen3 CONFIG QUIET)
if(Eigen3_FOUND)
    target_sources(unit_tests_core PRIVATE test_map_eigen.cpp)
    target_link_libraries(unit_tests_core PRIVATE Eigen3::Eigen)
endif()

find_package(OpenCV CONFIG QUIET)
if(OpenCV_FOUND)
    target_sources(unit_tests_core PRIVATE test_map_opencv.cpp)
    target_link_libraries(unit_tests_core PRIVATE ${OpenCV_LIBS})
endif()
