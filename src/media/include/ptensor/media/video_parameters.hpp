#pragma once

#include <string>

#include "time/rational.hpp"

namespace p10::media {
class VideoCodec {
  public:
    enum CodecType { H264, Other };

    VideoCodec() = default;

    VideoCodec(const CodecType& type) : type_(type) {}

    VideoCodec(const std::string& codec_name) {
        if (codec_name == "H264") {
            type_ = CodecType::H264;
        } else {
            type_ = CodecType::Other;
            codec_name_ = codec_name;
        }
    }

    CodecType type() const {
        return type_;
    }

    std::string to_string() const {
        switch (type_) {
            case CodecType::H264:
                return "H264";
            case CodecType::Other:
            default:
                return codec_name_;
        }
    }

  private:
    CodecType type_ = CodecType::Other;
    std::string codec_name_;
};

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

    VideoCodec codec() const {
        return codec_;
    }

    int bit_rate() const {
        return bit_rate_;
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

    VideoParameters& codec(const VideoCodec& codec) {
        codec_ = codec;
        return *this;
    }

    VideoParameters& bit_rate(int bit_rate) {
        bit_rate_ = bit_rate;
        return *this;
    }

  private:
    int width_ = 0;
    int height_ = 0;
    Rational frame_rate_ = {1, 24};
    VideoCodec codec_;
    int bit_rate_ = 1000000;
};
}  // namespace p10::media