#pragma once
#include <cinttypes>

namespace p10::media {
class Rational {
  public:
    constexpr Rational() = default;

    constexpr Rational(int64_t numerator, int64_t denominator) :
        numerator_(numerator),
        denominator_(denominator) {}

    double to_double() const {
        return static_cast<double>(numerator_) / static_cast<double>(denominator_);
    }

    int64_t num() const {
        return numerator_;
    }

    int64_t den() const {
        return denominator_;
    }

    Rational inverse() const {
        return Rational {
            denominator_,
            numerator_

        };
    }

  private:
    int64_t numerator_ = 0;
    int64_t denominator_ = 1;
};

inline bool operator==(const Rational& lhs, const Rational& rhs) {
    return lhs.num() == rhs.num() && lhs.den() == rhs.den();
}
}  // namespace p10::media