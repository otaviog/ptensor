add_library(unit_tests_media OBJECT
    test_media_capture.cpp test_video_queue.cpp
)
target_include_directories(
    unit_tests_media PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/ptensor/media)
ptensor_target_options(unit_tests_media Media)
target_link_libraries(unit_tests_media
    PUBLIC ptensor ptensor_media PRIVATE Catch2::Catch2 ptensor_testing)
