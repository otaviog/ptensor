#pragma once

#include <cstdint>

#include <ptensor/tensor.hpp>

#include "time/time.hpp"

namespace p10::media {
class AudioFrame {
  public:
    AudioFrame() = default;

    AudioFrame(Tensor&& samples, size_t sample_rate, const Time& time = Time()) :
        samples_(std::move(samples)),
        sample_rate_hz_(sample_rate),
        time_(time) {}

    double duration_seconds() const {
        return static_cast<double>(samples_count()) / sample_rate_hz_;
    }

    int64_t samples_count() const {
        return samples_.shape(1).unwrap();
    }

    int64_t channels_count() const {
        return samples_.shape(0).unwrap();
    }

    size_t sample_rate() const {
        return sample_rate_hz_;
    }

    void set_sample_rate(size_t new_sample_rate) {
        sample_rate_hz_ = new_sample_rate;
    }

    Time time() const {
        return time_;
    }

    void set_time(const Time& new_time) {
        time_ = new_time;
    }

    const Tensor& samples() const {
        return samples_;
    }

    Tensor& samples() {
        return samples_;
    }

  private:
    Tensor samples_;
    size_t sample_rate_hz_ = 0;
    Time time_;
};

}  // namespace p10::media