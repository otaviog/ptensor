add_executable(bench_core bench_transpose.cpp)
ptensor_target_options(bench_core "Core")
# The per-kernel benchmarks include the transpose kernel headers (src/core) and
# the simd internals (ptensor links simd PRIVATE, so the path is not inherited).
target_include_directories(bench_core PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/simd/include)
target_link_libraries(bench_core PRIVATE ptensor benchmark::benchmark Threads::Threads)
# bench instantiates the tile2d pragma itself; ptensor hides simd's OpenMP behind
# a PRIVATE link, so depend on it directly to compile/link the parallel path.
if(OpenMP_CXX_FOUND)
    target_link_libraries(bench_core PRIVATE OpenMP::OpenMP_CXX)
endif()
