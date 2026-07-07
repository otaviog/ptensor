add_executable(unit_tests_all)
ptensor_target_options(unit_tests_all "Testing")

target_sources(unit_tests_all PRIVATE
    $<TARGET_OBJECTS:unit_tests_core>
    $<TARGET_OBJECTS:unit_tests_testing>
    $<TARGET_OBJECTS:unit_tests_simd>
    $<TARGET_OBJECTS:unit_tests_log>)

target_link_libraries(unit_tests_all PRIVATE Catch2::Catch2WithMain ptensor_log)

# unit_tests_core objects may reference OpenCV symbols (map adapter tests);
# object libraries do not propagate their link deps through TARGET_OBJECTS.
find_package(OpenCV CONFIG QUIET)
if(OpenCV_FOUND)
    target_link_libraries(unit_tests_all PRIVATE ${OpenCV_LIBS})
endif()

if (WITH_IO)
    target_sources(unit_tests_all PRIVATE $<TARGET_OBJECTS:unit_tests_io>)
    target_link_libraries(unit_tests_all PRIVATE ptensor_io)
endif()

if (WITH_OP)
    target_sources(unit_tests_all PRIVATE $<TARGET_OBJECTS:unit_tests_op>)
    target_link_libraries(unit_tests_all PRIVATE ptensor_op)
endif()

if (WITH_MEDIA)
    target_link_libraries(unit_tests_all PRIVATE unit_tests_media)
endif()

if (WITH_INFER)
    target_link_libraries(unit_tests_all PRIVATE unit_tests_infer)
endif()

if (WITH_RECOG)
  target_link_libraries(unit_tests_all PRIVATE unit_tests_recog)
endif()

if (WITH_VIZ)
    target_link_libraries(unit_tests_all PRIVATE unit_tests_viz)
endif()

catch_discover_tests(unit_tests_all
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
