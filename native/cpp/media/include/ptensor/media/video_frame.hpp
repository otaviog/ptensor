#pragma once
#include <cstdint>

#include <ptensor/tensor.hpp>
#include <sys/types.h>

#include "time/time.hpp"

namespace p10::media {
class VideoFrame {
  public:
    size_t width() const {
        return image.shape(1).unwrap();
    }

    size_t height() const {
        return image.shape(0).unwrap();
    }

    size_t channels() const {
        return image.shape(2).unwrap();
    }

    Time get_time() const {
        return time;
    }

    std::span<uint8_t> as_bytes() {
        return std::span<uint8_t>(
            reinterpret_cast<uint8_t*>(image.as_bytes().data()),
            image.size_bytes()
        );
    }

    std::span<const uint8_t> as_bytes() const {
        return std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(image.as_bytes().data()),
            image.size_bytes()
        );
    }

  private:
    Tensor image;
    Time time;
};
}  // namespace p10::media