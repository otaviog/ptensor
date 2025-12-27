add_executable(sample_window window.cpp)
ptensor_target_options(sample_window "private/guiapp")

target_link_libraries(sample_window PRIVATE
    ptensor_guiapp
)
