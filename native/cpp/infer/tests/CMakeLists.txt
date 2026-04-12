add_library(unit_tests_infer OBJECT
    test_infer.cpp
)
target_include_directories(
    unit_tests_infer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/ptensor/infer)
ptensor_target_options(unit_tests_infer Infer)
target_link_libraries(unit_tests_infer
    PUBLIC ptensor ptensor_infer PRIVATE Catch2::Catch2 ptensor_testing)
