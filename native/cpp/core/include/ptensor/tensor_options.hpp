#pragma once

#include "axis.hpp"
#include "device.hpp"
#include "dtype.hpp"
#include "shape.hpp"

namespace p10 {

class TensorOptions {
  public:
    TensorOptions() = default;

    TensorOptions(Dtype dtype) : dtype_(dtype) {}

    TensorOptions(Dtype::Value dtype) : dtype_(dtype) {}

    /// The data type of the tensor.
    Dtype dtype() const {
        return dtype_;
    }

    /// The stride of the tensor.
    const Stride& stride() const {
        return stride_;
    }

    /// The axes of the tensor.
    const Axes& axes() const {
        return axes_;
    }

    /// The device of the tensor.
    Device device() const {
        return device_;
    }

    /// Sets the device of the tensor.
    TensorOptions& device(const Device& device) {
        device_ = device;
        return *this;
    }

    /// Sets the data type of the tensor.
    TensorOptions& dtype(Dtype dtype) {
        dtype_ = dtype;
        return *this;
    }

    /// Sets the stride of the tensor.
    TensorOptions& stride(const Stride& stride) {
        stride_ = stride;
        return *this;
    }

    /// Sets the axes of the tensor.
    TensorOptions& axes(const Axes& axes) {
        axes_ = axes;
        return *this;
    }

    TensorOptions clone() {
        return *this;
    }

  protected:
    Device device_ = Device::Cpu;
    Dtype dtype_ = Dtype::Float32;
    Stride stride_;
    Axes axes_;
};

template<typename scalar_t>
class MakeViewOptions {
  public:
    /// The device of the tensor.
    Device device() const {
        return device_;
    }

    /// Sets the device of the tensor.
    MakeViewOptions& device(const Device& device) {
        device_ = device;
        return *this;
    }

    /// The stride of the tensor.
    const Stride& stride() const {
        return stride_;
    }

    /// Sets the stride of the tensor.
    MakeViewOptions& stride(const Stride& stride) {
        stride_ = stride;
        return *this;
    }

    /// The axes of the tensor.
    const Axes& axes() const {
        return axes_;
    }

    /// Sets the axes of the tensor.
    MakeViewOptions& axes(const Axes& axes) {
        axes_ = axes;
        return *this;
    }

    TensorOptions to_options() const {
        return TensorOptions().stride(stride()).axes(axes()).dtype(Dtype::from<scalar_t>());
    }

  private:
    Stride stride_;
    Axes axes_;
    Device device_ = Device::Cpu;
};

}  // namespace p10
