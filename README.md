# PTensor — A Portable Tensor Library

A simple tensor library designed to be portable across languages and operating systems.

The main goal is to support edge deployment of Machine Learning (ML) and Computer Vision (CV) algorithms by:

* Removing the need to expose third-party libraries in public APIs. Consumers see only `p10::Tensor`; the implementation is free to use Eigen, OpenCV, NumPy, PyTorch, TFLite, etc.
  ```cpp
  // public header
  #include <ptensor/tensor.hpp>

  void process(p10::Tensor& input);

  // private implementation
  #include <other_library.hpp>

  void process(p10::Tensor& input) {
      auto matrix = as_eigen_matrix(input).unwrap();
      auto mat    = as_opencv_mat(input).unwrap();
      auto ten    = as_ort_tensor(input).unwrap();
  }
  ```

* Being agnostic to inference runtimes such as ONNX Runtime, OpenVINO, PyTorch, and TFLite.
  ```cpp
  #include <ptensor/infer/infer.hpp>
  #include <ptensor/recog/face_detection.hpp>

  auto detector = IFaceDetector::create(
      BlazeFaceModel(),
      infer::IInfer::from_onnx("<model_path>.onnx")
  ).unwrap();
  ```

* Providing essential operations for ML and CV pipelines in C++.
  ```cpp
  #include <ptensor/op/blur.hpp>
  #include <ptensor/op/fft.hpp>

  auto blurred = op::gaussian_blur(input, 5).unwrap();
  auto spectrum = op::fft(signal).unwrap();
  ```

* Supporting language bindings so users can prototype and benchmark deployments with Python or TypeScript.
  ```python
  import numpy as np
  import ptensor

  array = run_my_numpy_pipeline()
  tensor = ptensor.Tensor.from_numpy(array)

  result = run_my_deployment_code(tensor)
  output = result.numpy()

  evaluate(output)
  ```

* Serving as a proof-of-concept for AI deployment on edge architectures, including a WebAssembly build target.


## Building

First, fetch submodules if you haven't already:

```bash
git submodule update --init --recursive
```

### Linux (Ubuntu) and macOS

Linux dependencies:

```bash
sudo apt install git cmake ninja-build clang clang-tools clang-format autoconf libxcb1
```

macOS dependencies:

```bash
brew install ninja pkg-config
```

Debug build:

```bash
cmake --workflow --preset clang/debug/install
```

Release build:

```bash
cmake --workflow --preset clang/release/install
```

### Windows

```powershell
winget install -e --id Ninja-build.Ninja
winget install -e --id Kitware.CMake
cmake --workflow --preset msbuild/install
```

Clang also works on Windows:

```powershell
cmake --workflow --preset clang/debug/install
```

### WebAssembly

Install the Emscripten SDK first:

```bash
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk && ./emsdk install latest && ./emsdk activate latest
source ./emsdk_env.sh
```

Then build:

```bash
cmake --workflow --preset wasm/build
```

The output lands in `build/wasm/release/js/` as `ptensor.js` + `ptensor.wasm`.


## Using as a vcpkg Port

PTensor ships a vcpkg port under `ports/ptensor/`. You can consume it from any project that uses vcpkg.

### 1. Add a custom registry in `vcpkg-configuration.json`

Point vcpkg at the PTensor repository so it can find the port:

```json
{
  "default-registry": {
    "kind": "git",
    "baseline": "<vcpkg-baseline-commit>",
    "repository": "https://github.com/microsoft/vcpkg"
  },
  "registries": [
    {
      "kind": "git",
      "baseline": "<ptensor-commit-sha>",
      "repository": "https://github.com/otaviog/ptensor.git",
      "packages": ["ptensor"]
    }
  ]
}
```

### 2. Declare the dependency in `vcpkg.json`

```json
{
  "name": "your-project",
  "version": "1.0.0",
  "dependencies": [
    "ptensor"
  ]
}
```

Optional features can be enabled individually:

```json
{
  "dependencies": [
    {
      "name": "ptensor",
      "features": ["io", "op"]
    }
  ]
}
```

Available features:

| Feature | Description | Extra dependencies |
|---------|-------------|-------------------|
| `io` | Image and NumPy file I/O | `zlib`, `stb`, `audiofile` |
| `op` | Additional tensor operations (FFT, etc.) | `pocketfft` |
| `media` | Video and audio capture/write via FFmpeg | `ffmpeg` |
| `infer` | ONNX model inference | `onnxruntime` |

### 3. Link in `CMakeLists.txt`

```cmake
find_package(ptensor CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE ptensor::ptensor)
```

### 4. Basic usage

```cpp
#include <ptensor/tensor.hpp>
#include <ptensor/dtype.hpp>

int main() {
    auto shape  = p10::make_shape({2, 3, 4}).unwrap();
    auto tensor = p10::Tensor::zeros(p10::Dtype::from<float>(), shape);
    // pass tensor to your library or inference runtime
    return 0;
}
```

## LLM Usage

PTensor is heavily developed with [Claude Code](https://claude.ai/code). The split between fully AI-generated and human-guided work is intentional and worth documenting.

### Fully AI-generated

Areas where the code is written almost entirely by the LLM, with humans providing only direction:

* **CMake** — Once the initial module layout was established, maintenance is mostly mechanical. The LLM handles adding new targets, presets, and install rules without much supervision.
* **Language bindings** — After deciding on a C API with opaque pointers as the FFI boundary, the binding code in Python and TypeScript can be generated and maintained reliably. Key human decisions were: the initial Python binding architecture, the TypeScript `Tensor` API design, and the choice of Bun as the runtime.
* **WASM build** — The entire WebAssembly platform target (`platforms/wasm/`) is vibe-coded.
* **Documentation** — API doc comments and this README are revised by the LLM.

### AI-assisted (human-led)

Areas where LLMs are used for review, suggestions, bug hunting, and test generation, but human judgment drives the actual writing and outcome:

* C++ modules' architectures, code, and API design
* C-API architecture and core code
* `p10::op` algorithm development
* SIMD optimizations (AVX2 kernels in `src/simd/`)
* Unit test architecture and coverage


