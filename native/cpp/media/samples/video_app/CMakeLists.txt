add_executable(sample_video_app main.cpp
    imgui_impl_sdl2.h imgui_impl_sdl2.cpp
    imgui_impl_vulkan.h imgui_impl_vulkan.cpp)
ptensor_target_options(sample_video_app Media)

find_package(Vulkan REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(SDL2 CONFIG REQUIRED)
find_package(CLI11 CONFIG REQUIRED)

target_link_libraries(sample_video_app PRIVATE
    ptensor_media
    ptensor_io
    Vulkan::Vulkan
    imgui::imgui
    CLI11::CLI11
    $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
    $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
)
if (MSVC)
    set_target_properties(
        sample_video_app PROPERTIES
        VS_DEBUGGER_COMMAND_ARGUMENTS ${CMAKE_SOURCE_DIR}/tests/data/video/file_example_MP4_480_1_5MG.mp4
        VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endif()