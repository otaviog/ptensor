#pragma once

#include <array>
#include <string>

#include <ptensor/config.h>

#include <string_view>

#include "p10_result.hpp"

namespace p10 {
enum class AxisUsage { Any, Width, Height, Channel, Batch };

class Axis {
  public:
    Axis() = default;

    Axis(const std::string_view& name, AxisUsage usage) : name_(name), usage_(usage) {}

    explicit Axis(AxisUsage usage) : usage_(usage) {}

    explicit Axis(const std::string_view& name) : name_(name) {}

    const std::string& name() const {
        return name_;
    }

    AxisUsage usage() const {
        return usage_;
    }

  private:
    std::string name_;
    AxisUsage usage_ = AxisUsage::Any;
};

class Axes {
  public:
    Axes() = default;

    explicit Axes(int64_t dims) : dims_(dims) {}

    Axes(const std::array<Axis, P10_MAX_SHAPE>& axes, int64_t dims) : axes_(axes), dims_(dims) {}

    P10Result<Axis> operator[](size_t index) const {
        if (axes_.size() <= index) {
            return Err(P10Error::OutOfRange);
        }
        return Ok(Axis(axes_[index]));
    }

    int64_t dims() const {
        return dims_;
    }

    bool empty() const {
        return dims_ == 0;
    }

  private:
    std::array<Axis, P10_MAX_SHAPE> axes_;
    int64_t dims_ = 0;
};
}  // namespace p10
