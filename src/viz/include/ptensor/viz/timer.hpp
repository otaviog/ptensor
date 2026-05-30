#pragma once

#include <chrono>
#include <optional>

#include <ptensor/media/time/time.hpp>

namespace p10::viz {

/// Template timer for measuring elapsed time with pause/resume support.
template<typename clock = std::chrono::system_clock>
class Timer {
  public:
    /// Start or resume the timer.
    void start() {
        if (!start_) {
            start_ = clock::now();
        }

        if (pause_) {
            paused_duration_ += clock::now() - *pause_;
            pause_ = std::nullopt;
        }
    }

    /// Pause the timer without resetting it.
    void pause() {
        pause_ = clock::now();
    }

    /// Get the elapsed time since the timer started, excluding paused periods.
    p10::media::Time elapsed(p10::media::Rational base_time) const {
        if (!start_) {
            return p10::media::Time {base_time, 0};
        }

        if (pause_) {
            return calc_elapsed(base_time, *pause_);
        }
        return calc_elapsed(base_time, clock::now());
    }

    /// Reset the timer to initial state.
    void reset() {
        start_ = std::nullopt;
        pause_ = std::nullopt;
        paused_duration_ = typename clock::duration {0};
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

/// SystemClock-based timer alias.
using SystemTimer = Timer<std::chrono::system_clock>;

}  // namespace p10::viz