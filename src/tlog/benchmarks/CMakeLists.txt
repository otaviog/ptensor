add_executable(bench_tlog bench_tlog.cpp)
ptensor_target_options(bench_tlog "TLog")
# The bench times the stages under the public API: the session/connection
# headers live in the module dir, base64 and the zstd staging in src/core.
target_include_directories(bench_tlog PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tlog
    ${CMAKE_SOURCE_DIR}/src/core)
target_link_libraries(bench_tlog PRIVATE ptensor ptensor_tlog benchmark::benchmark Threads::Threads)
