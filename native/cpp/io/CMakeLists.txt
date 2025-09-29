add_library(hp3d_io STATIC)
target_sources(hp3d_io
  PRIVATE
  ${_INCLUDE_DIR}/io/numpy.hpp src/io/numpy.cpp
  ${_INCLUDE_DIR}/io/image.hpp src/io/image.cpp)
target_include_directories(hp3d_io PRIVATE ${_INCLUDE_DIR}/io)
target_link_libraries(hp3d_io hp3d cnpy stb_image)
