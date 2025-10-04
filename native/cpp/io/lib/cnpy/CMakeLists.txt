add_library(cnpy STATIC 
    cnpy.cpp
    cnpy.h)

target_include_directories(cnpy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cnpy PUBLIC ZLIB::ZLIB)
set_target_properties(cnpy PROPERTIES
    FOLDER Libraries)
