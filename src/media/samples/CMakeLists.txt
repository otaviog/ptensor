add_executable(sample_transcoder transcoder.cpp)
ptensor_target_options(sample_transcoder "Media")
target_link_libraries(sample_transcoder PRIVATE ptensor_media CLI11::CLI11)

add_executable(sample_media_player player.cpp)
ptensor_target_options(sample_media_player "Media")
target_link_libraries(sample_media_player PRIVATE ptensor_media ptensor_viz CLI11::CLI11)
set_target_properties(sample_media_player PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
