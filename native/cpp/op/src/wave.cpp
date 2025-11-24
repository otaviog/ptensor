#include "wave.hpp"

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
    const double phase = params.phaseRadians();

    const size_t period_samples = static_cast<size_t>(
        params.period().has_value() ? params.period().value() * sample_rate
                                    : (sample_rate / frequency)
    );
    if (period_samples == 0) {
        return P10Error::InvalidArgument << "Period results in zero samples.";
    }
    output.visit([&](auto span) {
        using scalar_t = typename decltype(span)::value_type;

        const scalar_t amplitude_val = static_cast<scalar_t>(amplitude);
        const scalar_t phase_val = static_cast<scalar_t>(phase);

        const scalar_t phase_increment =
            static_cast<scalar_t>(2.0 * M_PI * frequency / sample_rate);
        for (size_t i = 0; i < span.size(); ++i) {
            const scalar_t t = static_cast<scalar_t>(i % period_samples);
            const scalar_t normalized_phase = phase_increment * t;
            span[i] = amplitude_val * std::sin(normalized_phase + phase_val);
        }
    });

    return P10Error::Ok;
}
}  // namespace p10::op