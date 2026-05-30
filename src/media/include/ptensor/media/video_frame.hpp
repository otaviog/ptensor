#pragma once
#include <cstdint>

#include <ptensor/tensor.hpp>

#include "ptensor/dtype.hpp"
#include "ptensor/p10_error.hpp"
#include "ptensor/shape.hpp"
#include "time/time.hpp"

namespace p10::media {
/// Supported pixel format.
enum class PixelFormat {
    RGB24,  ///< 3 bytes per pixel
};

/// Video frame with image data and timing.
class VideoFrame {
  public:
    VideoFrame() = default;

    /// Create video frame from image tensor and optional time.
    VideoFrame(Tensor&& image, const Time& time = Time()) : image_(std::move(image)), time_(time) {}

    /// Create frame with specified dimensions and format.
    P10Error create(size_t width, size_t height, PixelFormat format) {
        if (format != PixelFormat::RGB24) {
            return P10Error(P10Error::InvalidArgument, "Unsupported pixel format");
        }
        return image_.create(make_shape(height, width, 3), Dtype::Uint8);
    }

    /// Get supported pixel format.
    static PixelFormat pixel_format() {
        return PixelFormat::RGB24;
    }

    /// Update frame timing.
    void update_time(const Time& time) {
        time_ = time;
    }

    /// Get width in pixels.
    size_t width() const {
        if (image_.empty()) {
            return 0;
        }
        return image_.shape(1).unwrap();
    }

    /// Get height in pixels.
    size_t height() const {
        if (image_.empty()) {
            return 0;
        }
        return image_.shape(0).unwrap();
    }

    /// Get number of channels.
    size_t channels() const {
        if (image_.empty()) {
            return 0;
        }
        return image_.shape(2).unwrap();
    }

    /// Get frame timing.
    Time time() const {
        return time_;
    }

    /// Get mutable byte span of image data.
    std::span<uint8_t> as_bytes() {
        return std::span<uint8_t>(
            reinterpret_cast<uint8_t*>(image_.as_bytes().data()),
            image_.size_bytes()
        );
    }

    /// Get const byte span of image data.
    std::span<const uint8_t> as_bytes() const {
        return std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(image_.as_bytes().data()),
            image_.size_bytes()
        );
    }

    /// Get stride information.
    Stride stride() const {
        return image_.stride();
    }

    /// Get data type.
    Dtype dtype() const {
        return image_.dtype();
    }

    /// Get const image tensor.
    const Tensor& image() const {
        return image_;
    }

    /// Get mutable image tensor.
    Tensor& image() {
        return image_;
    }

  private:
    Tensor image_;
    Time time_;
};
}  // namespace p10::media