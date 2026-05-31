#pragma once
#include <cstdint>

namespace p10::media {
/// Rational number (numerator/denominator).
class Rational {
  public:
    constexpr Rational() = default;

    /// Create rational from numerator and denominator.
    constexpr Rational(int64_t numerator, int64_t denominator) :
        numerator_(numerator),
        denominator_(denominator) {}

    /// Convert to double.
    double to_double() const {
        return static_cast<double>(numerator_) / static_cast<double>(denominator_);
    }

    /// Get numerator.
    int64_t num() const {
        return numerator_;
    }

    /// Get denominator.
    int64_t den() const {
        return denominator_;
    }

    /// Get inverted rational (den/num).
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

/// Check equality of two rationals.
inline bool operator==(const Rational& lhs, const Rational& rhs) {
    return lhs.num() == rhs.num() && lhs.den() == rhs.den();
}
}  // namespace p10::media