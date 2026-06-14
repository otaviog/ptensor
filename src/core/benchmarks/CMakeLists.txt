add_executable(bench_core bench_transpose.cpp)
ptensor_target_options(bench_core "Core")
# The per-kernel benchmarks include the transpose kernel headers (src/core) and
# the simd internals (ptensor links simd PRIVATE, so the path is not inherited).
target_include_directories(bench_core PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/simd/include)
target_link_libraries(bench_core PRIVATE ptensor benchmark::benchmark Threads::Threads)
