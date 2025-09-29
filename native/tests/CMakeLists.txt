# These tests can use the Catch2-provided main
add_executable(unit_tests_all)

target_sources(unit_tests_all PRIVATE $<TARGET_OBJECTS:unit_tests_core>)
target_link_libraries(unit_tests_all PRIVATE Catch2::Catch2WithMain ptensor)

catch_discover_tests(unit_tests_all
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
