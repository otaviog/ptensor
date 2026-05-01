add_library(unit_tests_log OBJECT test_log.cpp)
ptensor_target_options(unit_tests_log "Log")
target_link_libraries(unit_tests_log
    PUBLIC ptensor ptensor_log
    PRIVATE Catch2::Catch2 ptensor_testing)
