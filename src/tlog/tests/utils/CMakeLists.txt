add_executable(test_tlog_live
  test_tlog_live.cpp
)
ptensor_target_options(test_tlog_live "TLog")
target_link_libraries(test_tlog_live
    PUBLIC ptensor ptensor_tlog PRIVATE CLI11::CLI11)

# Manual driver for the VS Code tensor viewer: a program with tensors in scope
# at a breakpoint, so the extension has something to push.
add_executable(viewer_demo viewer_demo.cpp)
target_link_libraries(viewer_demo PRIVATE ptensor ptensor_tlog)
ptensor_target_options(viewer_demo "TLog")
set_target_properties(viewer_demo PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
