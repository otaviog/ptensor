add_executable(bench_core bench_transpose.cpp)
target_link_libraries(bench_core PRIVATE ptensor benchmark::benchmark benchmark::benchmark_main Threads::Threads)
