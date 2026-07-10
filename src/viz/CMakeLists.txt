find_package(SDL2 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

if (WIN32 OR LINUX)
  find_package(Vulkan REQUIRED)
  set(_USE_VULKAN_BACKEND ON)
elseif (APPLE)
  enable_language(OBJCXX)
  set(CMAKE_OBJCXX_STANDARD 20)
  set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
  set(_USE_METAL_BACKEND ON)
endif()

set(_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/ptensor/viz)
add_library(ptensor_viz STATIC)

set(PUBLIC_HEADERS
  ${_INCLUDE_DIR}/gui_app.hpp
  ${_INCLUDE_DIR}/gui_app_parameters.hpp
  ${_INCLUDE_DIR}/image_texture.hpp
  ${_INCLUDE_DIR}/timer.hpp
  ${_INCLUDE_DIR}/video_player_component.hpp
)

target_sources(ptensor_viz
  PUBLIC
    FILE_SET public_headers TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
    FILES ${PUBLIC_HEADERS}
  PRIVATE
    app_icon.cpp
    gui_app.cpp
    image_texture.cpp
    texture_stager.cpp
    video_player_component.cpp
)

if (_USE_VULKAN_BACKEND)
  target_sources(ptensor_viz
    PRIVATE
      vulkan/imgui_impl_sdl2.h vulkan/imgui_impl_sdl2.cpp
      vulkan/imgui_impl_vulkan.h vulkan/imgui_impl_vulkan.cpp
      vulkan/image_texture.impl.hpp vulkan/image_texture.impl.cpp
      vulkan/vulkan_error.hpp vulkan/vulkan_error.cpp
      vulkan/vulkan_context.hpp vulkan/vulkan_context.cpp
      vulkan/gui_app.impl.hpp vulkan/gui_app.impl.cpp
  )
  target_include_directories(ptensor_viz PRIVATE vulkan)
  set(_BACKEND_LIBS Vulkan::Vulkan)
  target_compile_definitions(ptensor_viz PRIVATE _USE_VULKAN_BACKEND)
elseif (_USE_METAL_BACKEND)
  target_sources(ptensor_viz
    PRIVATE
      vulkan/imgui_impl_sdl2.h vulkan/imgui_impl_sdl2.cpp
      metal/imgui_impl_metal.h metal/imgui_impl_metal.mm
      metal/image_texture.impl.hpp metal/image_texture.impl.cpp
      metal/gui_app.impl.hpp metal/gui_app.impl.cpp
  )
  # metal/ carries the Metal-CPP headers; vulkan/ provides the shared
  # imgui_impl_sdl2.h used by the Metal backend too.
  target_include_directories(ptensor_viz PRIVATE metal vulkan)
  set(_BACKEND_LIBS
    "-framework Metal"
    "-framework QuartzCore"
    "-framework Cocoa"
    "-framework Foundation"
  )
  # IMGUI_IMPL_METAL_CPP enables the C++ Metal-CPP API in imgui_impl_metal.mm
  target_compile_definitions(ptensor_viz PRIVATE _USE_METAL_BACKEND IMGUI_IMPL_METAL_CPP)
endif()

target_include_directories(ptensor_viz
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ptensor_viz
  PUBLIC
    ptensor
    ptensor_media
    imgui::imgui
  PRIVATE
    ptensor_log
    $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
    $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
    ${_BACKEND_LIBS}
)

ptensor_target_options(ptensor_viz "Viz")
if(_USE_METAL_BACKEND)
  target_compile_options(ptensor_viz PRIVATE -Wno-gnu-anonymous-struct)
endif()

if (BUILD_TESTS)
  add_subdirectory(tests)
endif()

install(TARGETS ptensor_viz
    ARCHIVE EXCLUDE_FROM_ALL
    FILE_SET public_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

