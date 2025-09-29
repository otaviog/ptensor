#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>

#include "axis.hpp"
#include "detail/blob.hpp"
#include "device.hpp"
#include "dtype.hpp"
#include "ptensor_result.hpp"
#include "shape.hpp"
#include "tensor_options.hpp"

namespace p10 {

using OptionalDeallocationFunction = std::optional<std::function<void(void*)>>;

/// Tensor class.
///
/// A tensor is a multi-dimensional array of elements of a single data type.
/// The tensor class provides a way to store and manipulate data in a
/// multi-dimensional array.
/// The tensor class can own its data or can be a view of another pointer data.
/// depending if the constructor froblob__view is used with a deallocation function.
class Tensor {
  public:
    /// Creates a tensor from a blob.
    ///
    /// # Arguments
    ///
    /// * `blob` - The blob that contains the tensor data.
    /// * `shape` - The shape of the tensor.
    /// * `options` - The tensor options.
    static Tensor from_data(
        void* data,
        const Shape& shape,
        const TensorOptions& options = TensorOptions(),
        const OptionalDeallocationFunction& dealloc = std::nullopt
    ) {
        return Tensor(Blob(data, options.device(), dealloc), shape, options);
    }

    template<typename scalar_t>
    static Tensor from_data(
        scalar_t* data,
        const Shape& shape,
        const MakeViewOptions<scalar_t>& view_options = MakeViewOptions<scalar_t>(),
        const OptionalDeallocationFunction& dealloc = std::nullopt
    ) {
        return from_data(data, shape, view_options.to_options(), dealloc);
    }

    /// Creates a tensor with zeros.
    ///
    /// # Arguments
    /// * `shape` - The shape of the tensor.
    /// * `options` - The tensor options.
    ///
    /// # Returns
    /// * A tensor with zeros. If the shape is empty, an empty tensor is returned.
    static PtensorResult<Tensor>
    zeros(const Shape& shape, const TensorOptions& options = TensorOptions()) {
        return full(shape, 0.0, options);
    }

    /// Creates a tensor with a given value.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `value` - The value to fill the tensor with.
    /// * `options` - The tensor options.
    ///
    /// # Returns
    /// * A tensor filled with the given value. If the shape is empty, an empty tensor is returned.
    static PtensorResult<Tensor> full(
        const Shape& shape,
        double value,
        const TensorOptions& options = TensorOptions()
    );

    static PtensorResult<Tensor>
    empty(const Shape& shape, const TensorOptions& options = TensorOptions());

    /// Default constructor. Creates an empty tensor.
    Tensor() = default;

    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;

    /// Checks if the tensor is empty.
    bool empty() const {
        return shape_.empty();
    }

    /// Gets the type of the tensor.
    Dtype dtype() const {
        return dtype_;
    }

    /// Gets the device of the tensor.
    Device device() const {
        return blob_.device();
    }

    /// Gets the number of dimensions of the tensor.
    size_t dims() const {
        return shape_.dims();
    }

    /// Gets the shape of the tensor the given axis.
    PtensorResult<int64_t> shape(size_t axis) const {
        return shape_[axis];
    }

    /// Gets the shape of the tensor.
    const Shape& shape() const {
        return shape_;
    }

    /// Gets the stride of the tensor.
    const Stride& stride() const {
        return stride_;
    }

    /// Gets the stride of the tensor at the given axis.
    PtensorResult<int64_t> stride(size_t axis) const {
        return stride_[axis];
    }

    /// Gets the number of elements of the tensor (i.e. sum of all dimensions)
    size_t size() const {
        return shape_.count();
    }

    /// Gets the size of the tensor in bytes.
    size_t size_bytes() const {
        return size() * dtype_.size();
    }

    /// Clones the tensor if the tensor is in CPU memory
    PtensorResult<Tensor> clone() const;

    Tensor as_view() {
        return Tensor(blob_.view(), shape(), options());
    }

    /// Returns the axes information.
    const Axes& axes() const {
        return axes_;
    }

    /// Returns a copy of the tensor options.
    TensorOptions options() const {
        return TensorOptions().dtype(dtype_).stride(stride_).axes(axes_).device(blob_.device());
    }

    template<typename scalar_t>
    constexpr std::span<const scalar_t> as_span1d() const {
        return std::span<const scalar_t>(blob_.data<scalar_t>(), size());
    }

    template<typename scalar_t>
    constexpr std::span<scalar_t> as_span1d() {
        return std::span<scalar_t>(blob_.data<scalar_t>(), size());
    }

    std::span<const std::byte> as_bytes() const {
        return std::span<const std::byte>(blob_.data<const std::byte>(), size_bytes());
    }

    std::span<std::byte> as_bytes() {
        return std::span<std::byte>(blob_.data<std::byte>(), size_bytes());
    }

    /// Returns true if the tensor is contiguous.
    bool is_contiguous() const;

    /// Returns a contiguous tensor.
    PtensorResult<Tensor> to_contiguous() const;

    void squeeze();

  private:
    Tensor(Blob&& blob, const Shape& shape, const TensorOptions& options) :
        blob_ {std::move(blob)},
        shape_ {shape} {
        set_options(options);
    }

    Tensor(const TensorOptions& options) {
        set_options(options);
    }

    void set_options(const TensorOptions& options) {
        dtype_ = options.dtype();
        stride_ = options.stride().empty() ? Stride(shape_) : options.stride();
        axes_ = options.axes().empty() ? Axes(shape_.dims()) : options.axes();
    }

    Blob blob_;
    Dtype dtype_;
    Shape shape_;
    Stride stride_;
    Axes axes_;
};
}  // namespace p10
