add_executable(bench_core bench_transpose.cpp)
ptensor_target_options(bench_core "Core")
target_link_libraries(bench_core PRIVATE ptensor benchmark::benchmark benchmark::benchmark_main Threads::Threads)
