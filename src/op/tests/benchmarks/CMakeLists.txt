add_executable(bench_op bench_blur.cpp)
ptensor_target_options(bench_op "Op")
# The per-kernel benchmarks include the private blur kernel header (src/op) and
# the simd internals it pulls in (ptensor links simd PRIVATE, so the path is not
# inherited).
target_include_directories(bench_op PRIVATE
    ${CMAKE_SOURCE_DIR}/src/op
    ${CMAKE_SOURCE_DIR}/src/simd/include)
target_link_libraries(bench_op PRIVATE ptensor_op benchmark::benchmark Threads::Threads)
