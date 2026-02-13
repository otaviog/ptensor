# Portable Tensor Docs Development Guide

* Project git is at: http://github.com/otaviog/ptensor.git

## Project Goals

* Provide a simple tensor library that's portable across different languages.
* Targets projects that what offer numerical and AI inference features without bringing in large dependencies.
* Do not aim to be near a full-featured tensor library like PyTorch or TensorFlow.
* Not have any external dependencies
* Support basic tensor operations (e.g., add, multiply, matmul, etc).
* Try to optimize with SIMD where possible
* Not tied to any specific backend (e.g., no CUDA, no OpenCL, etc).
* Easy to use C API for bindings.

## Build Commands

* Linux: `cmake --workflow --preset lin-dev`
* macOS: `cmake --workflow --preset mac-dev`
* Windows: `cmake --workflow --preset win-dev`
* WebAssembly: `cmake --workflow --preset wasm-build`

If vcpkg is not clone yet, run the following command first:

```shell
git submodule update --init --recursive
```

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
   cmake --workflow --preset wasm-build
   ```

The output will be in `build/wasm-release/` with files:
- `ptensor.js` - JavaScript loader
- `ptensor.wasm` - WebAssembly binary

### TypeScript/JavaScript Bindings

Two binding options are available:

1. **WebAssembly bindings** (for browsers and Node.js):
   - Located in `native/bindings/ptensor_bindings.cpp`
   - TypeScript wrappers in `bindings/typescript/ptensor-wrapper.ts`
   - Build with `cmake --workflow --preset wasm-build`
   - Use in browsers and Node.js

2. **FFI bindings** (for Node.js only):
   - Located in `bindings/typescript/src/`
   - Uses Koffi for native C API access
   - Requires built native library
   - Node.js only (better performance for server-side)

## Project Structure

* `src/` - Python source
* `cmake/` - CMake presets and modules
* `native/cpp` - C++ source
* `native/c` - C API
* `tests/` - All tests
* `ports` - vcpkg port files
* `vpkg.json` - Local vcpkg manifest

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