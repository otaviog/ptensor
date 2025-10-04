# Using PTensor as a vcpkg Port

This document explains how to use PTensor as a vcpkg port in other projects by pointing to your GitHub repository.

## Setup Instructions

### 1. Create a vcpkg registry pointing to your repository

In your consuming project, create a `vcpkg-configuration.json` file:

```json
{
  "default-registry": {
    "kind": "git",
    "baseline": "main",
    "repository": "https://github.com/Microsoft/vcpkg"
  },
  "registries": [
    {
      "kind": "git",
      "baseline": "main",
      "repository": "https://github.com/otaviog/ptensor.git",
      "packages": [ "ptensor" ]
    }
  ]
}
```

### 2. Add PTensor to your vcpkg.json

```json
{
  "name": "your-project",
  "version": "1.0.0",
  "dependencies": [
    "ptensor"
  ]
}
```

### 3. Use PTensor in your CMakeLists.txt

```cmake
find_package(ptensor CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE ptensor::ptensor)
```


## Features

PTensor supports the following vcpkg features:

- `io`: Enable I/O functionality (requires zlib and stb)
- `tests`: Build tests (for development, requires catch2)

Example with features:

```json
{
  "dependencies": [
    {
      "name": "ptensor",
      "features": [ "io" ]
    }
  ]
}
```
