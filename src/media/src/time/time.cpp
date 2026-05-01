#include "time/time.hpp"

namespace p10::media {
bool operator>(const Time& lhs, const Time& rhs) {
    if (lhs.base_.den() == rhs.base_.den()) {
        return lhs.stamp_ > rhs.stamp_;
    }

    if (lhs.base_.den() > rhs.base_.den()) {
        // is s1 * n1 / D1 > s2 * n2 / d2
        const int64_t factor = lhs.base_.den() / rhs.base_.den();
        return lhs.stamp_ > rhs.stamp_ * factor;
    } else {
        const int64_t factor = rhs.base_.den() / lhs.base_.den();
        return lhs.stamp_ * factor > rhs.stamp_;
    }
}
}  // namespace p10::media