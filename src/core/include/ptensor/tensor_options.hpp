#pragma once

#include "device.hpp"
#include "dtype.hpp"
#include "stride.hpp"

namespace p10 {

/// How a tensor's contents are meant to be interpreted. This is a hint only and
/// does not affect storage or layout. `Image` covers both a single image and a
/// batch of images (distinct from `Video`, a time sequence of frames).
enum class Usage : uint8_t { NotSpecified, Image, Signal, Audio, Video };

class TensorOptions {
  public:
    TensorOptions() = default;

    TensorOptions(Dtype dtype) : dtype_(dtype) {}

    TensorOptions(Dtype::Code dtype) : dtype_(dtype) {}

    /// The data type of the tensor.
    Dtype dtype() const {
        return dtype_;
    }

    /// The stride of the tensor.
    const Stride& stride() const {
        return stride_;
    }

    /// The usage hint of the tensor.
    Usage usage() const {
        return usage_;
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

    /// Sets the usage hint of the tensor.
    TensorOptions& usage(Usage usage) {
        usage_ = usage;
        return *this;
    }

    TensorOptions clone() {
        return *this;
    }

  protected:
    Device device_ = Device::Cpu;
    Dtype dtype_ = Dtype::Float32;
    Stride stride_;
    Usage usage_ = Usage::NotSpecified;
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

    /// The usage hint of the tensor.
    Usage usage() const {
        return usage_;
    }

    /// Sets the usage hint of the tensor.
    MakeViewOptions& usage(Usage usage) {
        usage_ = usage;
        return *this;
    }

    TensorOptions to_options() const {
        return TensorOptions().stride(stride()).usage(usage()).dtype(Dtype::from<scalar_t>());
    }

  private:
    Stride stride_;
    Usage usage_ = Usage::NotSpecified;
    Device device_ = Device::Cpu;
};

}  // namespace p10
