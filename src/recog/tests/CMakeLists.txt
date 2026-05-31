add_library(unit_tests_recog OBJECT
    test_face_detectors.cpp
    test_face_detection_to_string.cpp
)
target_include_directories(
    unit_tests_recog PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/ptensor/recog)
ptensor_target_options(unit_tests_recog Recog)
target_link_libraries(unit_tests_recog
    PUBLIC ptensor ptensor_recog PRIVATE Catch2::Catch2 ptensor_testing ptensor_io ptensor_op)
