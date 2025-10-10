set(_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/ptensor/op)
add_library(ptensor_op)

##
# Public Headers
set(PUBLIC_HEADERS
  ${_INCLUDE_DIR}/blur.hpp
  ${_INCLUDE_DIR}/elemwise.hpp
  ${_INCLUDE_DIR}/resize.hpp
  )

target_sources(ptensor_op
    PUBLIC 
    FILE_SET public_headers TYPE HEADERS
    FILES
    ${PUBLIC_HEADERS})
set_source_files_properties(${PUBLIC_HEADERS} PROPERTIES VS_FOLDER "Public Headers")

##
# Source to public headers
target_sources(ptensor_op
    PRIVATE
    src/blur.cpp
    src/elemwise.cpp
    src/resize.cpp
)

target_include_directories(ptensor_op
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE
    ${_INCLUDE_DIR})

target_link_libraries(ptensor_op PUBLIC ptensor)

ptensor_target_options(ptensor_op Op)

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

#### Installation
# Create alias for use in build tree and install
add_library(p10::op ALIAS ptensor_op)
include(GNUInstallDirs)
install(TARGETS ptensor_op
    EXPORT ptensorTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    FILE_SET public_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
