#pragma once

#include <chrono>
#include <cstdint>

#include "rational.hpp"

namespace p10::media {
/// Time with rational base (timebase).
class Time {
  public:
    Time() = default;

    /// Create time from base timebase and stamp.
    Time(Rational base, int64_t stamp) : base_(base), stamp_(stamp) {}

    /// Create time from seconds.
    static Time from_seconds(Rational base, double seconds) {
        return Time {base, static_cast<int64_t>(seconds * base.den()) / base.num()};
    }

    /// Create time from milliseconds.
    static Time from(Rational base, std::chrono::milliseconds milli) {
        return Time {base, static_cast<int64_t>(milli.count()) * base.den() / (base.num() * 1000)};
    }

    /// Get timestamp value.
    int64_t stamp() const {
        return stamp_;
    }

    /// Get timebase.
    Rational base() const {
        return base_;
    }

    /// Convert to seconds.
    double to_seconds() const {
        return static_cast<double>(stamp_) * base_.num() / base_.den();
    }

    /// Convert to a different timebase.
    Time into_base(Rational new_base) const {
        const int64_t new_stamp =
            stamp_ * new_base.den() * base_.num() / (new_base.num() * base_.den());
        return Time {new_base, new_stamp};
    }

    /// Compare times.
    friend bool operator>(const Time& lhs, const Time& rhs);

  private:
    Rational base_ = {0, 1};
    int64_t stamp_ = 0;
};
}  // namespace p10::media
