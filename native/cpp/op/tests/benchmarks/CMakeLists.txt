
find_package(Threads REQUIRED)
find_package(benchmark CONFIG REQUIRED)

add_executable(bench_op bench_blur.cpp)
target_link_libraries(bench_op PRIVATE ptensor_op benchmark::benchmark benchmark::benchmark_main Threads::Threads)