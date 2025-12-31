#pragma once

#include <chrono>
#include <cstdint>

#include "rational.hpp"

namespace p10::media {
class Time {
  public:
    Time() = default;

    Time(Rational base, int64_t stamp) : stamp_(stamp), base_(base) {}

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

  private:
    Rational base_ = {0, 1};
    int64_t stamp_ = 0;
};
}  // namespace p10::media