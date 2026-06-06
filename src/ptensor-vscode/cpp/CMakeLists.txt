# Manual test driver for the VS Code tensor viewer. Enabled with
# -DWITH_VSCODE_VIEWER_DEMO=ON. Links only the core ptensor library.
add_executable(vscode_viewer_demo viewer_demo.cpp)
target_link_libraries(vscode_viewer_demo PRIVATE ptensor)
ptensor_target_options(vscode_viewer_demo "Core")
set_target_properties(vscode_viewer_demo PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
