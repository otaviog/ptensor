#include <limits>

#include <ptensor/media/io/media_device.hpp>

namespace p10::media {

VideoParameters VideoDeviceInfo::match_closest(int width, int height, Rational frame_rate) const {
    if (capabilities_.empty()) {
        return VideoParameters{};
    }

    const VideoCapability* best = capabilities_.data();
    double best_score = std::numeric_limits<double>::max();

    const double fps_target =
        frame_rate.den() > 0 ? static_cast<double>(frame_rate.num()) / frame_rate.den() : 0.0;

    for (const auto& cap : capabilities_) {
        auto dw = static_cast<double>(cap.width() - width);
        auto dh = static_cast<double>(cap.height() - height);
        const double res_score = dw * dw + dh * dh;

        double fps_score = 0.0;
        if (fps_target > 0.0 && cap.max_frame_rate().den() > 0) {
            const double fps_cap = static_cast<double>(cap.max_frame_rate().num())
                / cap.max_frame_rate().den();
            const double df = fps_target - fps_cap;
            fps_score = df * df;
        }

        const double score = res_score + fps_score;
        if (score < best_score) {
            best_score = score;
            best = &cap;
        }
    }

    return VideoParameters{}
        .width(best->width())
        .height(best->height())
        .frame_rate(best->max_frame_rate());
}

AudioParameters AudioDeviceInfo::match_closest(double sample_rate, size_t num_channels) {
    if (capabilities_.empty()) {
        return AudioParameters{};
    }

    const AudioParameters* best = capabilities_.data();
    double best_score = std::numeric_limits<double>::max();

    for (const auto& cap : capabilities_) {
        double score = 0.0;

        if (sample_rate != SAMPLE_RATE_ANY) {
            const double d = cap.audio_sample_rate_hz() - sample_rate;
            score += d * d;
        }
        if (num_channels != NUMBER_OF_CHANNELS_ANY) {
            const double d =
                static_cast<double>(cap.audio_channels()) - static_cast<double>(num_channels);
            // Weight channels heavily — wrong channel count is usually a hard failure.
            score += d * d * 1e8;
        }

        if (score < best_score) {
            best_score = score;
            best = &cap;
        }
    }

    return *best;
}

}  // namespace p10::media
