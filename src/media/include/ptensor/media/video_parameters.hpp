#pragma once

#include <string>

#include "time/rational.hpp"

namespace p10::media {
/// Video codec type identifier.
class VideoCodec {
  public:
    /// Supported video codec types.
    enum CodecType { H264, Other };

    VideoCodec() = default;

    /// Create codec from type.
    VideoCodec(const CodecType& type) : type_(type) {}

    /// Create codec from name string.
    VideoCodec(const std::string& codec_name) {
        if (codec_name == "H264") {
            type_ = CodecType::H264;
        } else {
            type_ = CodecType::Other;
            codec_name_ = codec_name;
        }
    }

    /// Get codec type.
    CodecType type() const {
        return type_;
    }

    /// Get codec name as string.
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

/// Video capture and encoding parameters.
class VideoParameters {
  public:
    /// Get width in pixels.
    int width() const {
        return width_;
    }

    /// Get height in pixels.
    int height() const {
        return height_;
    }

    /// Get frame rate.
    Rational frame_rate() const {
        return frame_rate_;
    }

    /// Get video codec.
    VideoCodec codec() const {
        return codec_;
    }

    /// Get bit rate.
    int bit_rate() const {
        return bit_rate_;
    }

    /// Set width in pixels.
    VideoParameters& width(int width) {
        width_ = width;
        return *this;
    }

    /// Set height in pixels.
    VideoParameters& height(int height) {
        height_ = height;
        return *this;
    }

    /// Set frame rate.
    VideoParameters& frame_rate(const Rational& frame_rate) {
        frame_rate_ = frame_rate;
        return *this;
    }

    /// Set video codec.
    VideoParameters& codec(const VideoCodec& codec) {
        codec_ = codec;
        return *this;
    }

    /// Set bit rate.
    VideoParameters& bit_rate(int bit_rate) {
        bit_rate_ = bit_rate;
        return *this;
    }

    /// Get pixel format string.
    const std::string& pixel_format() const {
        return pixel_format_;
    }

    /// Set pixel format string.
    VideoParameters& pixel_format(std::string pixel_format) {
        pixel_format_ = std::move(pixel_format);
        return *this;
    }

  private:
    int width_ = 0;
    int height_ = 0;
    Rational frame_rate_ = {0, 1};  // 0 = unset; device/encoder picks the rate
    std::string pixel_format_ = "rgb";
    VideoCodec codec_;
    int bit_rate_ = 1000000;
};
}  // namespace p10::media
