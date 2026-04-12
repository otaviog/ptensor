#pragma once

namespace p10::recog {
template<typename T>
struct Point2 {
    Point2(T x = 0, T y = 0) : x(x), y(y) {}

    T x;
    T y;

    template<typename Q>
    Point2<Q> to() const {
        return Point2<Q> {static_cast<Q>(x), static_cast<Q>(y)};
    }
};

using Point2i = Point2<int>;
using Point2f = Point2<float>;
}  // namespace p10::recog
