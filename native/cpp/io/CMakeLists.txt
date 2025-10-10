find_package(ZLIB REQUIRED)
find_package(Stb REQUIRED)
add_subdirectory(lib/cnpy)

set(_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/ptensor/io)
add_library(ptensor_io STATIC)

set(PUBLIC_HEADERS
  ${_INCLUDE_DIR}/numpy.hpp
  ${_INCLUDE_DIR}/image.hpp
)

target_sources(ptensor_io
  PUBLIC
  FILE_SET public_headers TYPE HEADERS
  FILES ${PUBLIC_HEADERS}
  PRIVATE
  src/numpy.cpp
  src/image.cpp
)

target_include_directories(ptensor_io
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE
  ${_INCLUDE_DIR}
  ${Stb_INCLUDE_DIR}
)

target_link_libraries(ptensor_io PUBLIC ptensor PRIVATE $<BUILD_INTERFACE:cnpy> ZLIB::ZLIB)

ptensor_target_options(ptensor_io "IO")

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Create alias for use in build tree and install
add_library(p10::io ALIAS ptensor_io)

# Install the library and headers
install(TARGETS ptensor_io
    EXPORT ptensorTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    FILE_SET public_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
