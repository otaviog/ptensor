# Portable Tensor Docs Development Guide

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

* Linux / MacOS: `cmake --workflow --preset lin-dev`
* Windows: `cmake --workflow --preset win-dev`

If vcpkg is not clone yet, run the following command first:

```shell
git submodule update --init --recursive
```

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
