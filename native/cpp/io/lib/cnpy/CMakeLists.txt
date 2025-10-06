add_library(cnpy OBJECT
    cnpy.cpp
    cnpy.h)

target_include_directories(cnpy PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
target_link_libraries(cnpy PRIVATE ZLIB::ZLIB)
set_target_properties(cnpy PROPERTIES
    FOLDER Libraries
    POSITION_INDEPENDENT_CODE ON)
