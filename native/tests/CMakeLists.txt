add_executable(unit_tests_all)
ptensor_target_options(unit_tests_all "Testing")

target_sources(unit_tests_all PRIVATE 
    $<TARGET_OBJECTS:unit_tests_core>
    $<TARGET_OBJECTS:unit_tests_testing>
    $<TARGET_OBJECTS:unit_tests_simd>)

target_link_libraries(unit_tests_all PRIVATE Catch2::Catch2WithMain ptensor)

if (WITH_IO)
    target_sources(unit_tests_all PRIVATE $<TARGET_OBJECTS:unit_tests_io>)
    target_link_libraries(unit_tests_all PRIVATE ptensor_io)
endif()

if (WITH_OP)
    target_sources(unit_tests_all PRIVATE $<TARGET_OBJECTS:unit_tests_op>)
    target_link_libraries(unit_tests_all PRIVATE ptensor_op)
endif()

if (WITH_MEDIA)
    target_sources(unit_tests_all PRIVATE $<TARGET_OBJECTS:unit_tests_media>)
    target_link_libraries(unit_tests_all PRIVATE ptensor_media)
endif()

catch_discover_tests(unit_tests_all
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
