#pragma once

#include <chrono>
#include <cstdint>

#include "rational.hpp"

namespace p10::media {
class Time {
  public:
    Time() = default;

    Time(Rational base, int64_t stamp) : base_(base), stamp_(stamp) {}

    static Time from_seconds(Rational base, double seconds) {
        return Time {base, int64_t(seconds * base.den()) / base.num()};
    }

    static Time from(Rational base, std::chrono::milliseconds milli) {
        return Time {base, int64_t(milli.count()) * base.den() / (base.num() * 1000)};
    }

    int64_t stamp() const {
        return stamp_;
    }

    Rational base() const {
        return base_;
    }

    double to_seconds() const {
        return static_cast<double>(stamp_) * base_.num() / base_.den();
    }

    Time into_base(Rational new_base) const {
        int64_t new_stamp = stamp_ * new_base.den() * base_.num() / (new_base.num() * base_.den());
        return Time {new_base, new_stamp};
    }

    friend bool operator>(const Time& lhs, const Time& rhs);

  private:
    Rational base_ = {0, 1};
    int64_t stamp_ = 0;
};
}  // namespace p10::media
