#include "wave.hpp"

#include <numbers>

#include <ptensor/tensor.hpp>

#include "ptensor/p10_error.hpp"
#include "ptensor/shape.hpp"

namespace p10::op {
P10Error
generate_sine_wave(size_t num_samples, Dtype type, const SineWaveParams& params, Tensor& output) {
    P10_RETURN_IF_ERROR(output.create(make_shape(num_samples), type));

    const double frequency = params.frequency();
    const double sample_rate = params.sample_rate();

    if (frequency <= 0.0) {
        return P10Error::InvalidArgument << "Frequency must be positive.";
    }
    if (sample_rate <= 0.0) {
        return P10Error::InvalidArgument << "Sample rate must be positive.";
    }
    if (frequency > sample_rate / 2.0) {
        return P10Error::InvalidArgument
            << "Frequency must be less than or equal to half the sample rate (Nyquist frequency).";
    }
    const double amplitude = params.amplitude();
    const double phase = params.phase_radians();

    const auto period_samples = static_cast<size_t>(
        params.period().has_value() ? params.period().value() * sample_rate
                                    : (sample_rate / frequency)
    );
    if (period_samples == 0) {
        return P10Error::InvalidArgument << "Period results in zero samples.";
    }
    output.visit([&](auto span) {
        using scalar_t = decltype(span)::value_type;

        const auto amplitude_val = static_cast<scalar_t>(amplitude);
        const auto phase_val = static_cast<scalar_t>(phase);

        const auto phase_increment =
            static_cast<scalar_t>(2.0 * std::numbers::pi * frequency / sample_rate);
        for (size_t i = 0; i < span.size(); ++i) {
            const auto t = static_cast<scalar_t>(i % period_samples);
            const auto normalized_phase = phase_increment * t;
            span[i] = amplitude_val * std::sin(normalized_phase + phase_val);
        }
    });

    return P10Error::Ok;
}
}  // namespace p10::op
