add_library(unit_tests_viz OBJECT
    test_timer.cpp
)
target_link_libraries(unit_tests_viz
    PUBLIC ptensor ptensor_viz
    PRIVATE Catch2::Catch2 ptensor_testing
)
ptensor_target_options(unit_tests_viz Viz)

# Manual integration tests: they open a real window, so they are not
# registered with CTest; run the executables by hand.
add_executable(test_window test_window.cpp)
target_link_libraries(test_window PRIVATE
    ptensor_viz ptensor_io
)
ptensor_target_options(test_window "Viz")
set_target_properties(test_window PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_executable(test_video test_video.cpp)
target_link_libraries(test_video PRIVATE
    ptensor_viz ptensor_io ptensor_media
)
ptensor_target_options(test_video "Viz")
set_target_properties(test_video PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
