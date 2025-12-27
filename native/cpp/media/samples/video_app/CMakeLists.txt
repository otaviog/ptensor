add_executable(sample_video_app 
    main.cpp
    video_player_app.hpp video_player_app.cpp)
ptensor_target_options(sample_video_app Media)

find_package(CLI11 CONFIG REQUIRED)

target_link_libraries(sample_video_app PRIVATE
    ptensor_media
    ptensor_io
    ptensor_guiapp
    CLI11::CLI11)

if (MSVC)
    set_target_properties(
        sample_video_app PROPERTIES
        VS_DEBUGGER_COMMAND_ARGUMENTS ${CMAKE_SOURCE_DIR}/tests/data/video/file_example_MP4_480_1_5MG.mp4
        VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endif()
