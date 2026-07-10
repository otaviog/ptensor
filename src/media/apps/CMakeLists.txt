add_executable(transcoder transcoder.cpp)
ptensor_target_options(transcoder "Media")
target_link_libraries(transcoder PRIVATE ptensor_media CLI11::CLI11)

# Note: gate on WITH_VIZ, not `TARGET ptensor_viz` — the media module is
# configured before viz, so the target does not exist yet at this point.
if(WITH_VIZ)
  add_executable(media_player player.cpp)
  ptensor_target_options(media_player "Media")
  target_link_libraries(media_player PRIVATE ptensor_media ptensor_viz CLI11::CLI11)
  set_target_properties(media_player PROPERTIES
      VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )

  add_executable(camera_controller camera_controller.cpp)
  ptensor_target_options(camera_controller "Media")
  target_link_libraries(camera_controller PRIVATE
      ptensor_media ptensor_viz ptensor_io CLI11::CLI11
  )
  set_target_properties(camera_controller PROPERTIES
      VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )
endif()
