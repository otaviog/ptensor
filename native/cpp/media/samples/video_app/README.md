# PTensor Media Sample Application

A simple command-line application demonstrating video frame capture using the PTensor media library.

## Features

- Opens video files using FFmpeg
- Displays video metadata (resolution, frame rate)
- Reads and displays information about the first 5 frames

## Building

The sample is built automatically when `BUILD_SAMPLES` is enabled:

```bash
cmake --workflow --preset lin-dev
```

## Usage

```bash
./sample_video_app <video_file>
```

### Example

```bash
./sample_video_app tests/data/video/file_example_MP4_480_1_5MG.mp4
```

### Output

```
Opening video: tests/data/video/file_example_MP4_480_1_5MG.mp4

Video Information:
  Resolution: 480x270
  Frame rate: 30 fps

Reading first 5 video frames:
  Frame 0: 480x270 @ time=0.0s
  Frame 1: 480x270 @ time=0.033333s
  Frame 2: 480x270 @ time=0.066667s
  Frame 3: 480x270 @ time=0.1s
  Frame 4: 480x270 @ time=0.133333s

Video processed successfully!
```

## Notes

- The sample uses the PTensor MediaCapture API to read video frames
- Video frames are decoded to RGB24 format
- The application demonstrates basic error handling with P10Result

## GUI Version Limitations

A GUI version using ImGui + Vulkan was attempted but encountered dependency issues:

### GLFW Approach
- **Issue**: Requires X11 development libraries not readily available in WSL2
- **Required system packages**: `xorg-dev`, `libglu1-mesa-dev`, `xinerama`, `xcursor`, `pkg-config`
- **Install command**: `sudo apt-get install xorg-dev libglu1-mesa-dev`

### SDL2 Approach
- **Issue**: SDL2's dependency chain includes libxcrypt which requires GNU autotools
- **Required system packages**: `autoconf`, `autoconf-archive`, `automake`, `libtool`
- **Install command**: `sudo apt-get install autoconf autoconf-archive automake libtool`

### Recommendation
For a GUI-based video viewer sample, install the required system dependencies first:
- For GLFW: `sudo apt-get install xorg-dev libglu1-mesa-dev`
- For SDL2: `sudo apt-get install autoconf autoconf-archive automake libtool`

Then add the appropriate dependencies to `vcpkg.json` under the `samples` feature:
```json
"samples": {
  "description": "Build sample applications (for development)",
  "dependencies": ["imgui", "sdl2"]  // or "glfw3"
}
```
