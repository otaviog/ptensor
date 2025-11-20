#pragma once

#include <optional>
#include <ptensor/tensor.hpp>
#include "time/time.hpp"

namespace p10::media {
struct AudioFrame {

    Tensor samples;
    double sample_rate_hz = 0.0;
    std::optional<Time> start_time;

    double duration_seconds() const {
        return static_cast<double>(samples_count()) / sample_rate_hz;
    }

    int64_t samples_count() const {
        return samples.shape(1).unwrap();
    }

    int64_t channels_count() const {
        return samples.shape(0).unwrap();
    }
};

}