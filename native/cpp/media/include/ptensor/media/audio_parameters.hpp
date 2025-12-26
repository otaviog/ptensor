#pragma once

#include <cstddef>

namespace p10::media {
class AudioParameters {
  public:
    double audio_sample_rate_hz() const {
        return audio_sample_rate_hz_;
    }

    size_t audio_frame_size() const {
        return audio_frame_size_;
    }

    AudioParameters& audio_sample_rate_hz(double sample_rate_hz) {
        audio_sample_rate_hz_ = sample_rate_hz;
        return *this;
    }

    AudioParameters& audio_frame_size(size_t frame_size) {
        audio_frame_size_ = frame_size;
        return *this;
    }

  private:
    double audio_sample_rate_hz_ = 0.0;
    size_t audio_frame_size_ = 0;
};
}  // namespace p10::media