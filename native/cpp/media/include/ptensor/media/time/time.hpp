#pragma once

#include <cstdint>

#include "rational.hpp"

namespace p10::media {
class Time {
  public:
    Time() = default;

    Time(int64_t stamp, Rational base) : stamp_(stamp), base_(base) {}

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
    int64_t stamp_ = 0;
    Rational base_ = {0, 1};
};
}  // namespace p10::media