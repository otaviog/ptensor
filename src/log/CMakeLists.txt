find_package(spdlog CONFIG REQUIRED)

set(_INCLUDE_DIR include/p10_internal/log)
add_library(ptensor_log STATIC)

target_sources(ptensor_log
    PRIVATE
        log.cpp
    PUBLIC FILE_SET public_headers
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
        FILES
            ${_INCLUDE_DIR}/log.hpp
)

target_include_directories(ptensor_log
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(ptensor_log
    PUBLIC ptensor
    PRIVATE spdlog::spdlog)

if(WIN32)
    target_link_libraries(ptensor_log PRIVATE dbghelp)
endif()

set_target_properties(ptensor_log PROPERTIES FOLDER "Log")

install(TARGETS ptensor_log
    ARCHIVE EXCLUDE_FROM_ALL
    FILE_SET public_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if (BUILD_TESTS)
    add_subdirectory(tests)
endif()
