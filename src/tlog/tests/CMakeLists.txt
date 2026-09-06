add_library(unit_tests_tlog OBJECT
  mock_connection.hpp
    test_tlog.cpp
    test_session.cpp
    test_tcp_connection.cpp
    test_spin_lock.cpp
)
ptensor_target_options(unit_tests_tlog "TLog")
target_link_libraries(unit_tests_tlog
    PUBLIC ptensor ptensor_tlog PRIVATE Catch2::Catch2 ptensor_testing)

# The session and connection tests exercise the module private headers.
target_include_directories(unit_tests_tlog PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_subdirectory(utils)
