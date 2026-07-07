---
name: test-dev
description: "How to write and fix C++ unit tests in ptensor: Catch2 layout, naming, P10Error/P10Result matchers, tensor comparison, and running tests."
---

# Writing C++ unit tests

Use this skill when writing or fixing unit tests for any C++ module under
`src/<module>/`. Tests use [Catch2](https://github.com/catchorg/Catch2) and live
next to the code in `src/<module>/tests/test_<thing>.cpp`.

## Layout and naming

* One `TEST_CASE` per behavior. Name it `"<Module>::<class|feature>"` and put
  finer-grained behavior in `SECTION`s. Tags go in the second argument, e.g.
  `TEST_CASE("Core::Tensor", "[core][tensor]")`.
* Keep the test in the module's namespace (`namespace p10::<module>`), with
  file-local helpers in an anonymous `namespace { ... }`.
* `SECTION` for variations that share setup; the body before the sections reruns
  per section, so put fixtures there.

## Assertions

* `REQUIRE` / `CHECK` for plain conditions (`REQUIRE` aborts the case, `CHECK`
  continues).
* For anything returning `P10Error` or `P10Result`, use `REQUIRE_THAT` with the
  matchers in `p10::testing` from `<ptensor/testing/catch2_assertions.hpp>`:

  ```cpp
  #include <ptensor/testing/catch2_assertions.hpp>

  REQUIRE_THAT(tensor.create(make_shape(2, 3)), testing::is_ok());
  REQUIRE_THAT(bad_call(), testing::is_error(P10Error::InvalidArgument));
  ```

  Note the names are lowercase: `testing::is_ok()` and
  `testing::is_error(...)`. `unwrap()` a `P10Result` only after asserting it is
  ok (or when a failure should abort the test anyway).

## Comparing tensors

Do not loop element-by-element to compare two tensors — use
`p10::testing::compare_tensors` from `<ptensor/testing/compare_tensors.hpp>`. It
returns a `P10Error`, so pair it with `is_ok()`:

```cpp
#include <ptensor/testing/compare_tensors.hpp>

REQUIRE_THAT(testing::compare_tensors(output, expected), testing::is_ok());
```

* It checks shape, stride, and dtype before values, with a clear error message
  on mismatch.
* Float/double comparisons use a `1e-6` tolerance; override with
  `CompareOptions().tolerance(t)`. Integer dtypes compare exactly.
* When the expected result is itself produced by code, build it with a simple
  reference implementation and compare against the optimized path — see
  `src/simd/tests/test_tile.cpp` (scalar reference vs tiled SIMD).

## Avoiding duplication across dtypes

When the same test should cover several dtypes, drive it with `GENERATE` +
`DYNAMIC_SECTION` instead of copy-pasting:

```cpp
auto dtype = GENERATE(Dtype::Float32, Dtype::Int32, Dtype::UInt8);
DYNAMIC_SECTION("dtype " << dtype) {
    // test body using dtype
}
```

## Registering a new test file

Add the file to the module's test target in `src/<module>/tests/CMakeLists.txt`:

```cmake
add_library(unit_tests_<module> OBJECT test_existing.cpp test_new.cpp)
target_link_libraries(unit_tests_<module>
    PUBLIC ptensor_<module>_ ptensor ptensor_testing
    PRIVATE Catch2::Catch2)
```

`ptensor_testing` provides the matchers and `compare_tensors`.

## Running tests

From the repo root, build then run via CTest with the build preset for your
system (`clang/debug`, `clang/release`, `clang/debug-asan`, `msbuild/*`,
`wasm/*`):

```bash
just test                              # full suite, default preset
ctest --preset clang/debug             # full suite
ctest --preset clang/debug -R Simd     # only matching test names
ctest --preset clang/debug --rerun-failed
```

Build first if sources changed: `cmake --build build/clang/debug`.

## Reference

* `src/core/tests/test_tensor.cpp` — broad examples (sections, generators,
  matchers).
* `src/testing/include/ptensor/testing/catch2_assertions.hpp` — matcher
  definitions.
* `src/testing/include/ptensor/testing/compare_tensors.hpp` — comparison API.
