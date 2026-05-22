---
name: ptensor-new-module
description: "Scaffold a new module under src/<name> in the ptensor project, following the existing module layout (CMake target, public headers, sources, tests, and parent wiring)."
---

# Create a new ptensor module

Use this skill when the user asks for a new module under `src/`. Confirm the module name and library kind before scaffolding; otherwise act on the conventions below.

## Inputs to gather

Ask only if not given:

- **name** (required) — short, lower_case (e.g. `dsp`, `geom`). Used for directory, target `ptensor_<name>`, namespace `p10::<name>`, and include path `ptensor/<name>/`.
- **kind** — one of:
  - `static` (default) — compiled `.cpp` files, like `op` or `io`.
  - `interface` — header-only, like `simd`.
- **gate** — whether the module is optional (`WITH_<NAME>` CMake option, like `op`/`io`/`media`) or always built (like `core`/`simd`). Default: optional for new modules unless the user says otherwise.
- **first symbol** — optional starter header/source pair (e.g. `hello`). Skip if the user only wants the skeleton.

## Layout to produce

For a `static` module called `<name>`:

```
src/<name>/
  CMakeLists.txt
  include/ptensor/<name>/<header>.hpp     # public headers
  src/<source>.cpp                         # private implementation
  tests/
    CMakeLists.txt
    test_<source>.cpp
```

For an `interface` (header-only) module:

```
src/<name>/
  CMakeLists.txt
  include/ptensor/<name>/<header>.hpp
  tests/
    CMakeLists.txt
    test_<header>.cpp
```

## File templates

### `src/<name>/CMakeLists.txt` — static library

Mirror `src/io/CMakeLists.txt`. Replace `<name>` and capitalize for the folder label.

```cmake
set(_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/ptensor/<name>)
add_library(ptensor_<name> STATIC)

set(PUBLIC_HEADERS
  ${_INCLUDE_DIR}/<header>.hpp
)

target_sources(ptensor_<name>
  PUBLIC
  FILE_SET public_headers TYPE HEADERS
  BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
  FILES ${PUBLIC_HEADERS}
  PRIVATE
  src/<source>.cpp
)

target_include_directories(ptensor_<name>
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE
  ${_INCLUDE_DIR}
)

target_link_libraries(ptensor_<name> PUBLIC ptensor)

ptensor_target_options(ptensor_<name> "<Name>")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

add_library(p10::<name> ALIAS ptensor_<name>)

install(TARGETS ptensor_<name>
  EXPORT ptensorTargets
  LIBRARY DESTINATION ${PTENSOR_INSTALL_LIB_DIR}
  ARCHIVE DESTINATION ${PTENSOR_INSTALL_LIB_DIR}
  RUNTIME DESTINATION ${PTENSOR_INSTALL_BIN_DIR}
  FILE_SET public_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
```

### `src/<name>/CMakeLists.txt` — interface (header-only)

Mirror `src/simd/CMakeLists.txt`:

```cmake
set(_INCLUDE_DIR include/ptensor/<name>)
add_library(ptensor_<name> INTERFACE)

target_sources(ptensor_<name> INTERFACE FILE_SET public_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
    FILES
    ${_INCLUDE_DIR}/<header>.hpp
)

target_include_directories(ptensor_<name> INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

set_target_properties(ptensor_<name> PROPERTIES FOLDER "<Name>")

if (BUILD_TESTS)
    add_subdirectory(tests)
endif()
```

### `src/<name>/tests/CMakeLists.txt`

```cmake
add_library(unit_tests_<name> OBJECT
    test_<source>.cpp
)
ptensor_target_options(unit_tests_<name> "<Name>")
target_link_libraries(unit_tests_<name>
    PUBLIC ptensor ptensor_<name> PRIVATE Catch2::Catch2 ptensor_testing)
```

### `src/<name>/include/ptensor/<name>/<header>.hpp` — starter

```cpp
#pragma once

#include <ptensor/p10_result.hpp>

namespace p10::<name> {

P10Error <fn>();

}  // namespace p10::<name>
```

### `src/<name>/src/<source>.cpp` — starter

```cpp
#include "<header>.hpp"

namespace p10::<name> {

P10Error <fn>() {
    return P10Error::Ok;
}

}  // namespace p10::<name>
```

### `src/<name>/tests/test_<source>.cpp` — starter

```cpp
#include <catch2/catch_test_macros.hpp>

#include <ptensor/<name>/<header>.hpp>
#include <ptensor/testing/catch2_assertions.hpp>

using namespace p10;

TEST_CASE("<name>::<fn> returns Ok", "[<name>]") {
    REQUIRE_THAT(<name>::<fn>(), testing::is_ok());
}
```

## Wire-up edits

### `src/CMakeLists.txt`

For an **optional** module, add the gate near the existing block and the `add_subdirectory` block:

```cmake
set(WITH_<NAME> OFF CACHE BOOL "Build <Name> library")
...
if (WITH_<NAME>)
  add_subdirectory(<name>)
endif (WITH_<NAME>)
```

For an **always-built** module, add `add_subdirectory(<name>)` near `core`/`simd`/`testing` (no gate).

### `tests/CMakeLists.txt`

For an optional module, append a guarded block matching the others:

```cmake
if (WITH_<NAME>)
    target_sources(unit_tests_all PRIVATE $<TARGET_OBJECTS:unit_tests_<name>>)
    target_link_libraries(unit_tests_all PRIVATE ptensor_<name>)
endif()
```

For always-built, add `$<TARGET_OBJECTS:unit_tests_<name>>` to the existing `target_sources(unit_tests_all PRIVATE …)` list.

## Conventions to honor

- Public namespace is always `p10::<name>` (lower_case). Private helpers go in an anonymous namespace inside the `.cpp`.
- Public header guard: `#pragma once` (project disables `portability-avoid-pragma-once`).
- Public types use `CamelCase`, methods/free functions `lower_case`, but `Ok`/`Err` factories and STL-style traits/iterators may keep their idiomatic case (the `.clang-tidy` is configured for `aNy_CasE` on those).
- Tests use Catch2 with `testing::is_ok` / `testing::is_error` matchers from `<ptensor/testing/catch2_assertions.hpp>`.
- Returns: prefer `P10Result<T>` for fallible operations, `P10Error` for void-returning fallible operations.
- New optional modules need both the `WITH_<NAME>` gate in `src/CMakeLists.txt` and the matching guarded block in `tests/CMakeLists.txt` — they must move together.

## Validation

After scaffolding:

1. `cmake --workflow --preset clang/debug` (or pass `-DWITH_<NAME>=ON` if optional and not wired into the preset).
2. `ctest --preset clang/debug -R "<name>"` to confirm the new module's tests run.

Stop and surface the error if the configure or build step fails — do not paper over a broken parent CMakeLists edit.
