find_package(ZLIB REQUIRED)
find_package(Stb REQUIRED)
add_subdirectory(lib/cnpy)

set(_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/ptensor/io)
add_library(ptensor_io STATIC)
target_sources(ptensor_io
  PRIVATE
  ${_INCLUDE_DIR}/numpy.hpp src/numpy.cpp
  ${_INCLUDE_DIR}/image.hpp src/image.cpp)
target_include_directories(ptensor_io PRIVATE ${_INCLUDE_DIR} ${Stb_INCLUDE_DIR})
target_link_libraries(ptensor_io PUBLIC ptensor PRIVATE cnpy)
