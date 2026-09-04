find_package(stduuid CONFIG REQUIRED)

set(_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/ptensor/tlog)
add_library(ptensor_tlog STATIC)

set(PUBLIC_HEADERS
  ${_INCLUDE_DIR}/tlog.hpp
)

target_sources(ptensor_tlog
  PUBLIC
  FILE_SET public_headers TYPE HEADERS
  BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
  FILES ${PUBLIC_HEADERS}
  PRIVATE
  spin_lock.hpp
  logging.hpp
  tlog.cpp
  session.hpp
  session.cpp
  connection.hpp
  tcp_connection.hpp
  tcp_connection.cpp
)

target_include_directories(ptensor_tlog
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE
  ${_INCLUDE_DIR}
)

target_link_libraries(ptensor_tlog
  PUBLIC ptensor
  PRIVATE ptensor_log stduuid)

ptensor_target_options(ptensor_tlog "TLog")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

# Install the library and headers. Like the other modules, the archive itself
# is EXCLUDE_FROM_ALL (only the aggregate libptensor ships) and the target is
# not exported: it links `ptensor`, which is not in an export set either.
install(TARGETS ptensor_tlog
    ARCHIVE EXCLUDE_FROM_ALL
    FILE_SET public_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
