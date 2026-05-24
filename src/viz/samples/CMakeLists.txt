add_executable(sample_viz_window sample_window.cpp)
target_link_libraries(sample_viz_window PRIVATE
    ptensor_viz ptensor_io
)
ptensor_target_options(sample_viz_window "Viz")
set_target_properties(sample_viz_window PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_executable(sample_viz_video sample_video.cpp)
target_link_libraries(sample_viz_video PRIVATE
    ptensor_viz ptensor_io ptensor_media
)
ptensor_target_options(sample_viz_video "Viz")
set_target_properties(sample_viz_video PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
