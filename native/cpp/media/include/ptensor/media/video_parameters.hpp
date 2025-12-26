#pragma once

#include "time/rational.hpp"

namespace p10::media {
class VideoParameters {
  public:
    int width() const {
        return width_;
    }

    int height() const {
        return height_;
    }

    Rational frame_rate() const {
        return frame_rate_;
    }

    VideoParameters& width(int width) {
        width_ = width;
        return *this;
    }

    VideoParameters& height(int height) {
        height_ = height;
        return *this;
    }

    VideoParameters& frame_rate(const Rational& frame_rate) {
        frame_rate_ = frame_rate;
        return *this;
    }

  private:
    int width_ = 0;
    int height_ = 0;
    Rational frame_rate_ = {1, 24};
};
}  // namespace p10::media