FROM ubuntu:latest

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ninja-build cmake git curl zip unzip tar pkg-config linux-libc-dev nasm \
    clang-20 clang-tools-20 libclang-rt-20-dev llvm-20 \
    autoconf autoconf-archive automake libtool \
    libx11-dev libxext-dev libwayland-dev wayland-protocols libxkbcommon-dev \
    python3 python3-venv ca-certificates just \
    perl \
    make \
    libdbus-1-dev \
    libltdl-dev \
    libgl-dev libgles-dev \
    libpulse-dev \
    libasound2-dev \
    && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-20 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-20 100 \
    && update-alternatives --install /usr/bin/clang-scan-deps clang-scan-deps /usr/bin/clang-scan-deps-20 100 \
    && ln -sf /usr/bin/ninja /usr/local/bin/ninja \
    && rm -rf /var/lib/apt/lists/*
