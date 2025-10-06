add_executable(unit_tests_all)

target_sources(unit_tests_all PRIVATE $<TARGET_OBJECTS:unit_tests_core> $<TARGET_OBJECTS:unit_tests_testing>)

if (WITH_IO)
    target_sources(unit_tests_all PRIVATE $<TARGET_OBJECTS:unit_tests_io>)
endif()

if (WITH_OP)
    target_sources(unit_tests_all PRIVATE $<TARGET_OBJECTS:unit_tests_op>)
endif()

target_link_libraries(unit_tests_all PRIVATE Catch2::Catch2WithMain ptensor)

if (WITH_IO)
    target_link_libraries(unit_tests_all PRIVATE ptensor_io)
endif()

catch_discover_tests(unit_tests_all
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
