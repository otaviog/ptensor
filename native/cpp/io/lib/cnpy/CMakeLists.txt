add_library(cnpy STATIC 
    ${CMAKE_SOURCE_DIR}/lib/cnpy/cnpy.cpp
    ${CMAKE_SOURCE_DIR}/lib/cnpy/cnpy.h)

target_include_directories(cnpy PUBLIC ${CMAKE_SOURCE_DIR}/lib/cnpy)
target_link_libraries(cnpy PUBLIC ZLIB::ZLIB)
set_target_properties(cnpy PROPERTIES
    FOLDER Libraries)
