#pragma once

#include <string>

namespace p10::media {
class AudioCodec {
  public:
    enum CodecType { AAC, Other };

    AudioCodec() = default;

    AudioCodec(const CodecType& type) : type_(type) {}

    AudioCodec(const std::string& codec_name) {
        if (codec_name == "AAC") {
            type_ = CodecType::AAC;
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
            case CodecType::AAC:
                return "AAC";
            case CodecType::Other:
            default:
                return codec_name_;
        }
    }

  private:
    CodecType type_ = CodecType::Other;
    std::string codec_name_;
};

class AudioParameters {
  public:
    double audio_sample_rate_hz() const {
        return audio_sample_rate_hz_;
    }

    size_t audio_frame_size() const {
        return audio_frame_size_;
    }

    AudioCodec codec() const {
        return codec_;
    }

    size_t bit_rate() const {
        return bit_rate_;
    }

    size_t audio_channels() const {
        return audio_channels_;
    }

    AudioParameters& audio_sample_rate_hz(double sample_rate_hz) {
        audio_sample_rate_hz_ = sample_rate_hz;
        return *this;
    }

    AudioParameters& audio_frame_size(size_t frame_size) {
        audio_frame_size_ = frame_size;
        return *this;
    }

    AudioParameters& codec(const AudioCodec& codec) {
        codec_ = codec;
        return *this;
    }

    AudioParameters& bit_rate(size_t bit_rate) {
        bit_rate_ = bit_rate;
        return *this;
    }

    AudioParameters& audio_channels(size_t channels) {
        audio_channels_ = channels;
        return *this;
    }

  private:
    double audio_sample_rate_hz_ = 0.0;
    size_t audio_frame_size_ = 0;
    AudioCodec codec_;
    size_t bit_rate_ = 128000;
    size_t audio_channels_ = 2;
};
}  // namespace p10::media