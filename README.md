# PTensor - A Portable Tensor Library

A simple tensor library designed to be portable between languages and operating systems. 

The main objective is to support on edge deploy of Machine Learning (ML) and Computer Vision (CV) algorithms by: 

* Removing the need to expose 3rdParty libraries in the public APIs and expose interfaces to Eigen, OpenCV, Numpy, Pytorh, TFLite, etc...
  ```c++
  // public header
  #include <ptensor/tensor.hpp>
  
  void function(p10::Tensor &input);
  // private implementation
  #include <other libraries>
  void function(p10::Tensor &input) {
     // Use your favorite matrix library
     auto matrix = as_eigen_matrix(input).unwrap();
     auto mat = as_opencv_mat(input).unwrap();
     auto ten = as_ort_tensor(input).unwrap();
  }
  ```
* Agnostic to inference runtime like OnnxRuntime, OpenVINO, Pytorch, TFLite.
  ```c++
  #include <ptensor/infer/infer.hpp>
  #include <ptensor/infer/infer_config.hpp>
  #include <ptensor/recog/face_detection.hpp>

  auto* detector = IFaceDetector::create(BlazeFaceModel(),
        infer::IInfer::from_onnx("<model_path>.onnx"),
        infer::InferConfig::ORT // infer::InferConfig::OPEN_VINO, infer::InferConfig::TFlite
      ).unwrap()
    ).unwrap();
  ```
* Support of essential operations for creating ML and CV algorithms in C++.
  ```
  #include <ptensor/op/blur.hpp>
  #include <ptensor/op/laplacian_pyramid.hpp>
  #include <ptensor/op/fft.hpp>
  
  
  ```
* Support bindings for other languages, so users can easily create scripts to measure the various performance of their deployment with Typescript (bun) or Python.
  ```python
  import numpy as np
  from ptensor import Tensor
    
  array = run_my_numpy_complex()
  Tensor.from_numpy(array)
  result = run_my_deploy_code(array).to_numpy()
    
  eval_my_deploy(result)
  ```
* Be a proof of concept for AI deploytment on the edge architectures, like WASM build prototype.


## Building

Get submodules if not done at clone

```bash
git submodule update --init --recursive
```

### Linux (Ubuntu) and MacOS


Linux dependencies:

```bash
sudo apt install git cmake ninja-build clang clang-tools clang-format autoconf libxcb1
```

MacOS dependencies

```bash
brew install ninja pkg-config
```


```bash
cmake --workflow --preset clang/debug
```

### Windows

```powershell
winget install -e --id Ninja-build.Ninja 
winget install -e --id Kitware.CMake
cmake --workflow --preset msbuild/install
```

Clang also works on Windows too

```powershell
cmake --workflow --preset clang/debug
```

## Using it as vcpkg port file

