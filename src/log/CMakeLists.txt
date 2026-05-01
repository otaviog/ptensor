find_package(ng-log CONFIG REQUIRED)

set(_INCLUDE_DIR include/ptensor/log)
add_library(ptensor_log INTERFACE)

target_sources(ptensor_log INTERFACE FILE_SET public_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
    FILES
    ${_INCLUDE_DIR}/log.hpp
)

target_include_directories(ptensor_log INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ptensor_log INTERFACE ptensor ng-log::ng-log)

set_target_properties(ptensor_log PROPERTIES FOLDER "Log")

add_library(p10::log ALIAS ptensor_log)

install(TARGETS ptensor_log
    EXPORT ptensorTargets
    FILE_SET public_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if (BUILD_TESTS)
    add_subdirectory(tests)
endif()
