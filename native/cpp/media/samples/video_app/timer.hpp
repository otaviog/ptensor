#pragma once

#include <chrono>

#include <ptensor/media/time/time.hpp>

template<typename clock = std::chrono::system_clock>
class Timer {
  public:
    void start() {
        if (!start_) {
            start_ = clock::now();
        }

        if (pause_) {
            paused_duration_ += clock::now() - *pause_;
            pause_ = std::nullopt;
        }
    }

    void pause() {
        pause_ = clock::now();
    }

    p10::media::Time elapsed(p10::media::Rational base_time) const {
        if (!start_) {
            return p10::media::Time {base_time, 0};
        }

        if (pause_) {
            return calc_elapsed(base_time, *pause_);
        }
        return calc_elapsed(base_time, clock::now());
    }

  private:
    p10::media::Time calc_elapsed(p10::media::Rational base_time, clock::time_point now) const {
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - (*start_ + paused_duration_)
        );

        return p10::media::Time::from(base_time, elapsed_ms);
    }

    std::optional<typename clock::time_point> start_;
    std::optional<typename clock::time_point> pause_;
    typename clock::duration paused_duration_ {0};
};