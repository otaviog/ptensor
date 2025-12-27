#pragma once
#include <cstdint>

#include <ptensor/tensor.hpp>
#include <sys/types.h>

#include "ptensor/dtype.hpp"
#include "ptensor/p10_error.hpp"
#include "ptensor/shape.hpp"
#include "time/time.hpp"

namespace p10::media {
enum class PixelFormat {
    RGB24,  // 3 bytes per pixel
};

class VideoFrame {
  public:
    VideoFrame() = default;

    VideoFrame(Tensor&& image, const Time& time = Time()) : image_(std::move(image)), time_(time) {}

    P10Error create(size_t width, size_t height, PixelFormat format) {
        if (format != PixelFormat::RGB24) {
            return P10Error(P10Error::InvalidArgument, "Unsupported pixel format");
        }
        return image_.create(make_shape(height, width, 3), Dtype::Uint8);
    }

    PixelFormat pixel_format() const {
        return PixelFormat::RGB24;
    }

    void update_time(const Time& time) {
        time_ = time;
    }

    size_t width() const {
        if (image_.empty()) {
            return 0;
        }
        return image_.shape(1).unwrap();
    }

    size_t height() const {
        if (image_.empty()) {
            return 0;
        }
        return image_.shape(0).unwrap();
    }

    size_t channels() const {
        if (image_.empty()) {
            return 0;
        }
        return image_.shape(2).unwrap();
    }

    Time get_time() const {
        return time_;
    }

    std::span<uint8_t> as_bytes() {
        return std::span<uint8_t>(
            reinterpret_cast<uint8_t*>(image_.as_bytes().data()),
            image_.size_bytes()
        );
    }

    std::span<const uint8_t> as_bytes() const {
        return std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(image_.as_bytes().data()),
            image_.size_bytes()
        );
    }

    Stride stride() const {
        return image_.stride();
    }

    Dtype dtype() const {
        return image_.dtype();
    }

    const Tensor& image() const {
        return image_;
    }
  private:
    Tensor image_;
    Time time_;
};
}  // namespace p10::media