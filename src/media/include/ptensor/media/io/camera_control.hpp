#pragma once

namespace p10::media {

/// Value range for a live-capture camera control (see
/// MediaCapture::get_brightness_range() and similar accessors).
class CameraControlRange {
  public:
    /// Get minimum value.
    int min() const {
        return min_;
    }

    /// Get maximum value.
    int max() const {
        return max_;
    }

    /// Get the smallest increment between valid values.
    int step() const {
        return step_;
    }

    /// Get the device's default value.
    int default_value() const {
        return default_value_;
    }

    /// Set minimum value.
    CameraControlRange& min(int value) {
        min_ = value;
        return *this;
    }

    /// Set maximum value.
    CameraControlRange& max(int value) {
        max_ = value;
        return *this;
    }

    /// Set step size.
    CameraControlRange& step(int value) {
        step_ = value;
        return *this;
    }

    /// Set default value.
    CameraControlRange& default_value(int value) {
        default_value_ = value;
        return *this;
    }

  private:
    int min_ = 0;
    int max_ = 0;
    int step_ = 1;
    int default_value_ = 0;
};

}  // namespace p10::media
