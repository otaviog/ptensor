# PTensor - A Portable Tensor Library

A simple tensor library designed to be portable between languages and operating systems.

## Build dev

```bash
$ sudo apt install git cmake ninja-build clang clang-tools clang-format autoconf libxcb1
```

```bash
$ brew install ninja pkg-config
```

```bash
$ git submodule update --init --recursive
### One of the following presets
# Linux dev
$ cmake --workflow --preset lin-dev
# macOS dev
$ cmake --workflow --preset mac-dev
# Win dev
$ cmake --workflow --preset win-dev
```
