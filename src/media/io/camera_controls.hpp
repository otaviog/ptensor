#pragma once

#include <memory>

#include <ptensor/media/io/camera_control.hpp>
#include <ptensor/p10_result.hpp>

namespace p10::media {

/// Identifies an integer-valued camera control.
enum class CameraControlId {
    Brightness,
    Contrast,
    Saturation,
    Gain,
    Zoom,
    FocusDistance,
    Exposure,
    WhiteBalanceTemperature
};

/// Identifies a boolean auto/manual camera control.
enum class CameraAutoControlId { Focus, Exposure, WhiteBalance };

/// Maps camera controls to a platform's device control API (V4L2 ioctl on
/// Linux, AVFoundation on macOS, DirectShow on Windows).
///
/// This is separate from the FFmpeg demuxer that streams frames: FFmpeg does
/// not expose the underlying device handle, so a backend opens its own
/// control-only handle to the same device.
class CameraControlBackend {
  public:
    virtual ~CameraControlBackend() = default;

    virtual P10Result<int> get(CameraControlId id) const = 0;
    virtual P10Error set(CameraControlId id, int value) = 0;
    virtual P10Result<CameraControlRange> get_range(CameraControlId id) const = 0;

    virtual P10Result<bool> get_auto(CameraAutoControlId id) const = 0;
    virtual P10Error set_auto(CameraAutoControlId id, bool enabled) = 0;
};

/// Open a control backend for the given video device index. Returns nullptr
/// (not an error) when the current platform has no backend implementation, or
/// when the device could not be opened for control access; callers fall back
/// to reporting NotImplemented/IoError on each individual accessor call.
std::unique_ptr<CameraControlBackend> open_camera_control_backend(int video_device_index);

}  // namespace p10::media
