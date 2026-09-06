add_executable(test_tlog_live
  test_tlog_live.cpp
)
ptensor_target_options(test_tlog_live "TLog")
target_link_libraries(test_tlog_live
    PUBLIC ptensor ptensor_tlog PRIVATE CLI11::CLI11)
