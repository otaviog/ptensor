#pragma once

#include <optional>

#include <ptensor/dtype.hpp>
#include <ptensor/p10_error.hpp>

namespace p10 {
class Tensor;
}  // namespace p10

namespace p10::op {
using Hz = double;
using Seconds = double;

class SineWaveParams {
  public:
    SineWaveParams& sample_rate(Hz rate) {
        sample_rate_ = rate;
        return *this;
    }

    SineWaveParams& frequency(Hz freq) {
        frequency_ = freq;
        return *this;
    }

    SineWaveParams& amplitude(double amp) {
        amplitude_ = amp;
        return *this;
    }

    SineWaveParams& phaseRadians(double ph) {
        phaseRadians_ = ph;
        return *this;
    }

    SineWaveParams& period(Seconds per) {
        period_ = per;
        return *this;
    }

    Hz sample_rate() const {
        return sample_rate_;
    }

    Hz frequency() const {
        return frequency_;
    }

    double amplitude() const {
        return amplitude_;
    }

    double phaseRadians() const {
        return phaseRadians_;
    }

    std::optional<Seconds> period() const {
        return period_;
    }

  private:
    Hz sample_rate_ = 44100.0;
    Hz frequency_ = 1.0;
    double amplitude_ = 1.0;
    double phaseRadians_ = 0.0;
    std::optional<Seconds> period_ = std::nullopt;
};

P10Error
generate_sine_wave(size_t num_samples, Dtype type, const SineWaveParams& params, Tensor& output);

}  // namespace p10::op