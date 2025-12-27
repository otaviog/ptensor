add_executable(sample_window window.cpp)
ptensor_target_options(sample_window "private/guiapp")

target_link_libraries(sample_window PRIVATE
    ptensor_guiapp ptensor_io
)
set_target_properties(sample_window PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
