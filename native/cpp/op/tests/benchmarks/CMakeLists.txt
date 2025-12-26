add_executable(bench_op bench_blur.cpp)
ptensor_target_options(bench_op "Op")
target_link_libraries(bench_op PRIVATE ptensor_op benchmark::benchmark benchmark::benchmark_main Threads::Threads)
