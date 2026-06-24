---
name: add-benchmark
description: "How to add C++ micro-benchmarks in ptensor: Google Benchmark layout, the module benchmarks/ target + CMake wiring, throughput reporting, kernel-level benches that bypass dispatch, and running in release."
---

# Adding C++ benchmarks

Use this skill when adding or changing performance benchmarks for a C++ module
under `src/<module>/`. Benchmarks use [Google Benchmark](https://github.com/google/benchmark)
(not Catch2 — that is for the unit tests) and live in
`src/<module>/benchmarks/bench_<thing>.cpp`.

Benchmarks measure speed only. **Correctness is the unit tests' job** — do not
add result-checking to a benchmark. A kernel that is fast but wrong is caught by
`src/<module>/tests/test_<thing>.cpp`, so make sure that coverage exists first.

## Layout

* Keep benches in the module namespace (`namespace p10::<module>` or `p10`),
  file-local in an anonymous `namespace { ... }`.
* One function per case, named `BM_<Thing>_<Variant>`, registered at the bottom
  with `BENCHMARK(...)`, and `BENCHMARK_MAIN();` at the very end.

```cpp
void BM_Transpose_Float32(benchmark::State& state) {
    const int64_t size = state.range(0);
    Tensor const input = make_input(size, size, Dtype::Float32);
    Tensor output;
    for (auto _ : state) {
        input.transpose(output);
        benchmark::DoNotOptimize(output);   // keep the result live
        benchmark::ClobberMemory();         // force the writes
    }
    const int64_t elements = size * size;
    state.SetItemsProcessed(state.iterations() * elements);
    state.SetBytesProcessed(state.iterations() * elements * sizeof(float));
}
BENCHMARK(BM_Transpose_Float32)->Arg(256)->Arg(1024)->Arg(2048)
    ->Unit(benchmark::kMicrosecond);
```

* Always end the inner loop with `DoNotOptimize` + `ClobberMemory`, or the
  compiler deletes the work.
* Report throughput with `SetItemsProcessed` / `SetBytesProcessed` so the
  output has `items_per_second` / `bytes_per_second` to compare across sizes.

## Minimal reuse

Bench bodies are nearly identical (build input, loop, report). Factor the shared
parts so a new case is a one-line wrapper:

* One `make_input(rows, cols, dtype)` helper, **fixed RNG seed** (e.g.
  `std::mt19937_64 const rng(42)`) so runs are reproducible. clang-tidy warns
  about the constant seed (`cert-msc51-cpp`); that is intentional here — ignore.
* One templated `run_<op><T>(state, ...)` helper that does the loop + reporting;
  the `BM_*` functions just call it with their dtype/shape.
* Keep a **naive untiled baseline** (`BM_*_Naive`) next to the real path to
  isolate the win (e.g. cache blocking vs SIMD).

## CMake wiring

Each module's `benchmarks/CMakeLists.txt` builds one `bench_<module>` exe.
Register it from the **module** `CMakeLists.txt` with
`add_subdirectory(benchmarks)`.

```cmake
add_executable(bench_core bench_transpose.cpp)
ptensor_target_options(bench_core "Core")
# ptensor links the simd lib PRIVATE, so its internal include path is NOT
# inherited; add it (and the module dir for sibling impl headers) explicitly.
# Use absolute ${CMAKE_SOURCE_DIR} paths, not relative ../.., so moving the
# benchmarks dir does not silently break the include resolution.
target_include_directories(bench_core PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/simd/include)
target_link_libraries(bench_core PRIVATE ptensor benchmark::benchmark Threads::Threads)
```

* You only need the extra `target_include_directories` when the bench includes
  private impl headers (`src/<module>/*.hpp`) or simd internals
  (`p10_internal/simd/*`). A bench that touches only the public `<ptensor/*>`
  API needs nothing beyond linking `ptensor`.
* The simd **symbols** (`l1_cache_size`, `is_supported`, ...) do link
  transitively (a static lib's PRIVATE deps reach the final exe); only the
  *include path* fails to propagate.

## Benching one kernel below the public API

To compare individual kernels (e.g. AVX2 vs NEON vs portable vs scalar) bypass
the cpuid dispatch in the public op and drive the tiler directly with each
kernel's `.fn`:

```cpp
simd::tile2d_autoblock<8, int32_t>(rows, cols, kernel.fn, border);
```

* Build the kernel spec from the same `make_*_transpose(...)` factories the
  production code uses, so you bench the real kernel.
* **Only register a kernel under its arch macro**, or you benchmark the empty
  stand-in kernel (a no-op that posts meaninglessly fast numbers):

  ```cpp
  #if PTENSOR_HAS_INTRINSICS_H
      BENCHMARK(BM_Kernel_Avx2)->Arg(256)->Arg(1024)->Unit(benchmark::kMicrosecond);
  #endif
  #if PTENSOR_HAS_NEON
      BENCHMARK(BM_Kernel_Neon)->Arg(256)->Arg(1024)->Unit(benchmark::kMicrosecond);
  #endif
  ```

## Running

Benchmarks must run from a **release** build — a debug build prints
`Library was built as DEBUG. Timings may be affected` and the numbers are
useless (intrinsics unoptimized).

```bash
cmake --workflow --preset clang/release
./build/clang/release/src/<module>/benchmarks/bench_<module>
# Filter to a subset:
./build/clang/release/src/core/benchmarks/bench_core --benchmark_filter="BM_Kernel"
```
