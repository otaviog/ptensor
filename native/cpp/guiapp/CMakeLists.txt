find_package(Vulkan REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(SDL2 CONFIG REQUIRED)
find_package(CLI11 CONFIG REQUIRED)

set(_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/ptensor/guiapp)
add_library(ptensor_guiapp STATIC)
ptensor_target_options(ptensor_guiapp "private/guiapp")

set(PUBLIC_HEADERS
  ${_INCLUDE_DIR}/gui_app.hpp
)

target_sources(ptensor_guiapp
  PUBLIC
  FILE_SET public_headers TYPE HEADERS
  BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
  FILES ${PUBLIC_HEADERS}
  PRIVATE
  src/gui_app.cpp
  src/gui_app.impl.hpp src/gui_app.impl.cpp
  src/imgui_impl_sdl2.h src/imgui_impl_sdl2.cpp
  src/imgui_impl_vulkan.h src/imgui_impl_vulkan.cpp
  src/vulkan_error.hpp src/vulkan_error.cpp
  src/vulkan_context.hpp src/vulkan_context.cpp
  )

target_include_directories(ptensor_guiapp
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
  ${_INCLUDE_DIR}
)

target_link_libraries(ptensor_guiapp
  PUBLIC
    ptensor
    imgui::imgui
  PRIVATE
    Vulkan::Vulkan
    $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
    $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
)

if(BUILD_SAMPLES)
  add_subdirectory(samples/window)
endif()
