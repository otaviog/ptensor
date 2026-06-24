# Portable Tensor Docs Development Guide

* Project git is at: http://github.com/otaviog/ptensor.git

## Project Goals

* Provide a simple tensor library that's portable across different languages.
* Targets projects that what offer numerical and AI inference features without bringing in large dependencies.
* Do not aim to be near a full-featured tensor library like PyTorch or TensorFlow.
* Not have any external dependencies
* Support basic tensor operations (e.g., add, multiply, image resize, fft, etc). Stuff needed for preprocessing. 
* Try to optimize with SIMD where possible (prefer AVX2)
* Not tied to any specific backend (e.g., no CUDA, no OpenCL, etc).
* Easy to use C API for bindings.

## Build Commands

* Linux/MacOs: `cmake --workflow --preset clang/debug` or `cmake --workflow --preset clang/release`
* Windows: `cmake --workflow --preset msbuild/install`
* WebAssembly: `cmake --workflow --preset wasm/build`

If vcpkg is not clone yet, run the following command first:

```shell
git submodule update --init --recursive
```

### OpenMP (parallel tiling)

`TileExecution::PARALLEL` in the simd tiler uses OpenMP. gcc (Linux) and MSVC
(Windows) ship the runtime, so no extra step there. Apple clang does not — on
macOS install libomp once:

```shell
brew install libomp
```

CMake finds it automatically (`find_package(OpenMP)`, with a brew hint on macOS).
If it is missing the build still works; `PARALLEL` just runs sequentially.

### WebAssembly Build

To build for WebAssembly, you need to have Emscripten SDK installed:

1. Install Emscripten:
   ```shell
   git clone https://github.com/emscripten-core/emsdk.git
   cd emsdk
   ./emsdk install latest
   ./emsdk activate latest
   source ./emsdk_env.sh
   ```

2. Build the project:
   ```shell
   cmake --workflow --preset wasm/build
   ```

The output will be in `build/wasm/release/` with files:
- `ptensor.js` - JavaScript loader
- `ptensor.wasm` - WebAssembly binary

### TypeScript/JavaScript Bindings

Two binding options are available:

1. **WebAssembly bindings** (for browsers and Node.js):
   - Located in `bindings/typescript/backends/wasm/`
   - TypeScript sources in `bindings/typescript/src/ptensor/`
   - Build with `cmake --workflow --preset wasm-build`
   - Use in browsers and Node.js

2. **FFI bindings** (for Node.js only):
   - Located in `bindings/typescript/src/`
   - Uses Koffi for native C API access
   - Requires built native library
   - Node.js only (better performance for server-side)

## Project Structure

* `src/` - C++ source (modules: `core`, `op`, `simd`, `io`, `media`, `infer`, `recog`, `testing`)
* `src/c/` - C API
* `bindings/python/` - Python bindings (sources in `bindings/python/src/ptensor/`)
* `bindings/typescript/` - TypeScript bindings (FFI + WASM backend in `backends/wasm/`)
* `cmake/` - CMake presets and modules
* `tests/` - Test data and outputs (per-module C++ tests live under `src/<module>/tests/`)
* `ports/` - vcpkg port files
* `vcpkg.json` - Local vcpkg manifest

## Test

To run unit test use from the root folder

```bash
just test
# Or
ctest --preset clang/debug # Or other config
```

To specific test, use:

```bash
ctest --preset clang/debug --R "<regular expression>"
```

To run failed only use:

```bash
ctest --preset clang/debug --rerun-failed
```

## Human Language

* English - default
* Don't make over the top statements, this is a simple library.

## Coding tips

* Use testing::IsError and testing::IsOk helpers for Result checks in tests. Like:
  ```cpp
    REQUIRE_THAT(
        some_function_call(),
        testing::IsError(P10Error::InvalidArgument)
    );
    ```
* When writing test for Dtype multiple times, use GENERATE and DYNAMIC_SECTION to avoid code duplication. Like:
  ```cpp
    auto dtype = GENERATE(Dtype::Float32, Dtype::Int32, Dtype::UInt8);
    DYNAMIC_SECTION("Testing with dtype " << dtype) {
        // test code here
    }
  ```
* Use clang-format for C/C++ code formatting. Run `clang-format -i <file>` to format a file in place.
