set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(lib/ng-log EXCLUDE_FROM_ALL)

set(_INCLUDE_DIR include/p10_internal/log)
add_library(ptensor_log STATIC)

target_sources(ptensor_log
    PRIVATE
        log.cpp
        # Embed ng-log objects directly — no exported link dependency
        $<TARGET_OBJECTS:ng-log_internal>
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
    PRIVATE
        $<TARGET_PROPERTY:ng-log_internal,INTERFACE_INCLUDE_DIRECTORIES>
)

target_compile_definitions(ptensor_log PRIVATE
    $<TARGET_PROPERTY:ng-log_internal,INTERFACE_COMPILE_DEFINITIONS>
)

target_link_libraries(ptensor_log PUBLIC ptensor)

set_target_properties(ptensor_log PROPERTIES FOLDER "Log")

install(TARGETS ptensor_log
    ARCHIVE EXCLUDE_FROM_ALL
    FILE_SET public_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if (BUILD_TESTS)
    add_subdirectory(tests)
endif()
