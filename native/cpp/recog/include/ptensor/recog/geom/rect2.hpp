#pragma once
#include <algorithm>

#include "point2.hpp"

namespace p10::recog {

template<typename T>
struct Rect2 {
    using Point = Point2<T>;

    Rect2(Point min = Point(), Point max = Point()) : min(min), max(max) {}

    Point2<T> min;
    Point2<T> max;

    T area() const {
        return (max.x - min.x) * (max.y - min.y);
    }

    T iou(const Rect2<T>& rhs) const {
        const auto left = std::max(min.x, rhs.min.x);
        const auto right = std::min(max.x, rhs.max.x);
        const auto top = std::max(min.y, rhs.min.y);
        const auto bottom = std::min(max.y, rhs.max.y);

        const auto intersection = std::max(right - left, T(0)) * std::max(bottom - top, T(0));
        return intersection / (area() + rhs.area() - intersection);
    }

    Rect2 scale(T x_scale, T y_scale) const {
        return Rect2(
            Point2(min.x * x_scale, min.y * y_scale),
            Point2(max.x * x_scale, max.y * y_scale)
        );
    }

    template<typename Q>
    Rect2<Q> to() const {
        return Rect2<Q> {min.template to<Q>(), max.template to<Q>()};
    }
};

using Rect2i = Rect2<int>;
using Rect2f = Rect2<float>;
}  // namespace p10::recog
