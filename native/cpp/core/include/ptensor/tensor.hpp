#pragma once

#include <cstdint>
#include <optional>
#include <random>

#include <type_traits>

#include "accessor3d.hpp"
#include "axis.hpp"
#include "detail/blob.hpp"
#include "detail/complex_traits.hpp"
#include "device.hpp"
#include "dtype.hpp"
#include "iterator.hpp"
#include "p10_error.hpp"
#include "p10_result.hpp"
#include "shape.hpp"
#include "span2d.hpp"
#include "span3d.hpp"
#include "stride.hpp"
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
    static P10Result<Tensor>
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
    static P10Result<Tensor>
    full(const Shape& shape, double value, const TensorOptions& options = TensorOptions());

    static P10Result<Tensor>
    empty(const Shape& shape, const TensorOptions& options = TensorOptions());

    static P10Result<Tensor> from_range(
        const Shape& shape,
        const TensorOptions& options = TensorOptions(),
        int64_t start = 0
    );

    static P10Result<Tensor> from_random(
        const Shape& shape,
        std::mt19937_64 rng,
        const TensorOptions& options = TensorOptions(),
        double min = 0.0,
        double max = 1.0
    );

    P10Error create(const Shape& shape, const TensorOptions& options = TensorOptions());

    P10Error create_like(const Tensor& other) {
        return create(other.shape(), other.options());
    }

    /// Clones the tensor if the tensor is in CPU memory
    P10Result<Tensor> clone() const;

    /// Default constructor. Creates an empty tensor.
    Tensor() = default;

    Tensor(Tensor&&);
    Tensor& operator=(Tensor&&);

    /// Checks if the tensor is empty (has zero dimensions or zero elements).
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
    P10Result<int64_t> shape(size_t axis) const {
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
    P10Result<int64_t> stride(size_t axis) const {
        return stride_[axis];
    }

    /// Gets the number of elements of the tensor (i.e. sum of all dimensions)
    size_t size() const {
        return shape_.count();
    }

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

    std::span<const std::byte> as_bytes() const {
        return std::span<const std::byte>(blob_.data<const std::byte>(), size_bytes());
    }

    std::span<std::byte> as_bytes() {
        return std::span<std::byte>(blob_.data<std::byte>(), size_bytes());
    }

    /// Gets the size of the tensor in bytes.
    size_t size_bytes() const {
        return size() * dtype_.size_bytes();
    }

    template<typename scalar_t>
    P10Result<Iterator<scalar_t>> iterator() {
        auto data_res = data_as<scalar_t>();
        if (!data_res.is_ok()) {
            return Err(data_res.error());
        }

        return Ok(Iterator<scalar_t> {
            data_res.unwrap(),
            shape_.as_span(),
            stride_.as_span(),
        });
    }

    template<typename scalar_t>
    P10Result<Iterator<const scalar_t>> iterator() const {
        auto data_res = data_as<scalar_t>();
        if (!data_res.is_ok()) {
            return Err(data_res.error());
        }

        return Ok(Iterator<const scalar_t> {
            data_res.unwrap(),
            shape_.as_span(),
            stride_.as_span(),
        });
    }

    template<typename scalar_t>
    P10Result<std::span<const scalar_t>> as_span1d() const {
        auto data_res = data_as<scalar_t>();
        if (!data_res.is_ok()) {
            return Err(data_res.error());
        }
        auto elem_count = size();
        if constexpr (detail::is_complex_v<std::remove_const_t<scalar_t>>) {
            elem_count /= 2;
        }
        return Ok(std::span<const scalar_t>(blob_.data<scalar_t>(), elem_count));
    }

    template<typename scalar_t>
    P10Result<std::span<scalar_t>> as_span1d() {
        auto data_res = data_as<scalar_t>();
        if (!data_res.is_ok()) {
            return Err(data_res.error());
        }
        auto elem_count = size();
        if constexpr (detail::is_complex_v<std::remove_const_t<scalar_t>>) {
            elem_count /= 2;
        }
        return Ok(std::span<scalar_t> {blob_.data<scalar_t>(), elem_count});
    }

    template<typename T>
    P10Result<Span2D<T>> as_span2d() {
        P10_RETURN_ERR_IF_ERROR(check_dims_for_2d_span<T>());
        auto data_res = data_as<T>();
        if (!data_res.is_ok()) {
            return Err(data_res.error());
        }
        const auto shape = shape_.as_span();
        return Ok(Span2D<T> {data_res.unwrap(), size_t(shape[0]), size_t(shape[1])});
    }

    template<typename T>
    P10Result<Span2D<const T>> as_span2d() const {
        P10_RETURN_ERR_IF_ERROR(check_dims_for_2d_span<T>());

        auto data_res = data_as<const T>();
        if (!data_res.is_ok()) {
            return Err(data_res.error());
        }
        const auto shape = shape_.as_span();
        return Ok(Span2D<const T> {data_res.unwrap(), size_t(shape[0]), size_t(shape[1])});
    }

    template<typename T>
    P10Result<Span3D<T>> as_span3d() {
        P10_RETURN_ERR_IF_ERROR(check_dims_for_3d_span<T>());

        auto data_res = data_as<T>();
        if (!data_res.is_ok()) {
            return Err(data_res.error());
        }
        const auto shape = shape_.as_span();
        return Ok(
            Span3D<T> {data_res.unwrap(), size_t(shape[0]), size_t(shape[1]), size_t(shape[2])}
        );
    }

    template<typename T>
    P10Result<Span3D<const T>> as_span3d() const {
        P10_RETURN_ERR_IF_ERROR(check_dims_for_3d_span<T>());

        auto data_res = data_as<const T>();
        if (!data_res.is_ok()) {
            return Err(data_res.error());
        }
        const auto shape = shape_.as_span();
        return Ok(Span3D<const T> {
            data_res.unwrap(),
            size_t(shape[0]),
            size_t(shape[1]),
            size_t(shape[2])
        });
    }

    template<typename T>
    P10Result<PlanarSpan3D<T>> as_planar_span3d() {
        P10_RETURN_ERR_IF_ERROR(check_dims_for_3d_span<T>());

        auto data_res = data_as<T>();
        if (!data_res.is_ok()) {
            return Err(data_res.error());
        }

        const auto shape = shape_.as_span();
        return Ok(PlanarSpan3D<T> {
            data_res.unwrap(),
            size_t(shape[0]),
            size_t(shape[1]),
            size_t(shape[2])
        });
    }

    template<typename T>
    P10Result<PlanarSpan3D<const T>> as_planar_span3d() const {
        P10_RETURN_ERR_IF_ERROR(check_dims_for_3d_span<T>());
        auto data_res = data_as<const T>();
        if (data_res.is_error()) {
            return Err(data_res.error());
        }
        const auto shape = shape_.as_span();
        return Ok(PlanarSpan3D<const T> {
            data_res.unwrap(),
            size_t(shape[0]),
            size_t(shape[1]),
            size_t(shape[2])
        });
    }

    template<typename T>
    P10Result<Accessor1D<T>> as_accessor1d() {
        P10_RETURN_ERR_IF_ERROR(check_dims_for_accessor1d<T>());

        auto data_res = data_as<T>();
        if (!data_res.is_ok()) {
            return Err(data_res.error());
        }

        return Ok(Accessor1D<T> {
            data_res.unwrap(),
            shape_[0].unwrap(),
            stride_[0].unwrap(),
        });
    }

    template<typename T>
    P10Result<Accessor1D<const T>> as_accessor1d() const {
        P10_RETURN_ERR_IF_ERROR(check_dims_for_accessor1d<T>());

        auto data_res = data_as<const T>();
        if (!data_res.is_ok()) {
            return Err(data_res.error());
        }

        return Ok(Accessor1D<const T> {
            data_res.unwrap(),
            shape_[0].unwrap(),
            stride_[0].unwrap(),
        });
    }

    template<typename T>
    P10Result<Accessor2D<T>> as_accessor2d() {
        P10_RETURN_ERR_IF_ERROR(check_dims_for_2d_access<T>());

        auto data_res = data_as<T>();
        if (!data_res.is_ok()) {
            return Err(data_res.error());
        }

        auto shape = shape_.as_span();
        auto stride = stride_.as_span();

        return Ok(Accessor2D<T> {
            data_res.unwrap(),
            std::array<int64_t, 2> {shape[0], shape[1]},
            std::array<int64_t, 2> {
                stride[0],
                stride[1],
            }
        });
    }

    template<typename T>
    P10Result<Accessor2D<const T>> as_accessor2d() const {
        P10_RETURN_ERR_IF_ERROR(check_dims_for_2d_access<T>());

        auto data_res = data_as<const T>();
        if (!data_res.is_ok()) {
            return Err(data_res.error());
        }

        auto shape = shape_.as_span();
        auto stride = stride_.as_span();

        return Ok(Accessor2D<const T> {
            data_res.unwrap(),
            std::array<int64_t, 2> {shape[0], shape[1]},
            std::array<int64_t, 2> {
                stride[0],
                stride[1],
            }
        });
    }

    template<typename T>
    P10Result<Accessor3D<T>> as_accessor3d() {
        P10_RETURN_ERR_IF_ERROR(check_dims_for_3d_access<T>());

        auto data_res = data_as<T>();
        if (!data_res.is_ok()) {
            return Err(data_res.error());
        }

        auto shape = shape_.as_span();
        auto stride = stride_.as_span();

        return Ok(Accessor3D<T> {
            data_res.unwrap(),
            std::array<int64_t, 3> {shape[0], shape[1], shape[2]},
            std::array<int64_t, 3> {
                stride[0],
                stride[1],
                stride[2],
            }
        });
    }

    template<typename T>
    P10Result<Accessor3D<const T>> as_accessor3d() const {
        P10_RETURN_ERR_IF_ERROR(check_dims_for_3d_access<T>());

        auto data_res = data_as<const T>();
        if (!data_res.is_ok()) {
            return Err(data_res.error());
        }

        auto shape = shape_.as_span();
        auto stride = stride_.as_span();

        return Ok(Accessor3D<const T> {
            data_res.unwrap(),
            std::array<int64_t, 3> {shape[0], shape[1], shape[2]},
            std::array<int64_t, 3> {
                stride[0],
                stride[1],
                stride[2],
            }
        });
    }

    auto visit(auto&& visitor) {
        return dtype_.visit(std::forward<decltype(visitor)>(visitor), as_bytes());
    }

    auto visit(auto&& visitor) const {
        return dtype_.visit(std::forward<decltype(visitor)>(visitor), as_bytes());
    }

    /// Returns true if the tensor is contiguous.
    bool is_contiguous() const {
        return is_contiguous_;
    }

    /// Returns a contiguous tensor.
    P10Result<Tensor> to_contiguous() const;

    void squeeze();

    P10Error unsqueeze(int64_t dim);

    P10Result<Tensor> select_dimension(int64_t dim, int64_t index);

    /// Reshapes the tensor to the given shape.
    /// # Arguments
    /// * `new_shape` - The new shape of the tensor.
    /// # Returns
    /// * An error if the reshape is not possible.
    P10Error reshape(const Shape& new_shape);

    /// Transposes a 2D tensor.
    /// # Arguments
    /// * `other` - The transposed tensor.
    /// # Returns
    /// * An error if the tensor is not 2D or not contiguous or device is not CPU.
    P10Error transpose(Tensor& other) const;

    P10Error fill(double value);

    P10Error copy_from(const Tensor& src);

  private:
    Tensor(Blob&& blob, const Shape& shape, const TensorOptions& options) :
        blob_ {std::move(blob)},
        shape_ {shape} {
        set_options(options);
    }

    Tensor(const TensorOptions& options) {
        set_options(options);
    }

    void set_options(const TensorOptions& options);

    template<typename T>
    P10Result<T*> data_as() {
        if (auto err = validate_dtype<T>(); !err.is_ok()) {
            return Err(err);
        }
        return Ok<T*>(blob_.data<T>());
    }

    template<typename T>
    P10Result<const T*> data_as() const {
        if (auto err = validate_dtype<T>(); !err.is_ok()) {
            return Err(err);
        }
        return Ok<const T*>(blob_.data<T>());
    }

    template<typename T>
    P10Error validate_dtype() const {
        if constexpr (detail::is_complex<std::remove_const_t<T>>::value) {
            if (dtype_ != Dtype::from<typename T::value_type>()) {
                return P10Error(P10Error::InvalidArgument, "Invalid dtype");
            }
            if (shape_.back() != 2) {
                return P10Error(
                    P10Error::InvalidArgument,
                    "Complex types require the last dimension to be of size 2"
                );
            }
        } else {
            if (dtype_ != Dtype::from<T>()) {
                return P10Error(P10Error::InvalidArgument, "Invalid dtype");
            }
        }
        return P10Error::Ok;
    }

    template<typename T>
    P10Error check_dims_for_accessor1d() const {
        if constexpr (detail::is_complex_v<std::remove_const_t<T>>) {
            if (dims() != 2) {
                return P10Error::InvalidArgument
                    << "Tensor must have 2 dimensions [N x 2] for complex types";
            }
        } else {
            if (dims() != 1) {
                return P10Error::InvalidArgument << "Tensor must have 1 dimension";
            }
        }
        return P10Error::Ok;
    }

    template<typename T>
    P10Error check_dims_for_2d_access() const {
        if constexpr (detail::is_complex_v<std::remove_const_t<T>>) {
            if (dims() != 3) {
                return P10Error::InvalidArgument
                    << "Tensor must have 3 dimensions [N x T x 2] for complex types";
            }
        } else {
            if (dims() != 2) {
                return P10Error::InvalidArgument << "Tensor must have 2 dimensions";
            }
        }
        return P10Error::Ok;
    }

    template<typename T>
    P10Error check_dims_for_3d_access() const {
        if constexpr (detail::is_complex_v<std::remove_const_t<T>>) {
            if (dims() != 4) {
                return P10Error::InvalidArgument
                    << "Tensor must have 4 dimensions [N x H x W x 2] for complex types";
            }
        } else {
            if (dims() != 3) {
                return P10Error::InvalidArgument << "Tensor must have 3 dimensions";
            }
        }
        return P10Error::Ok;
    }

    template<typename T>
    P10Error check_dims_for_2d_span() const {
        if (!is_contiguous()) {
            return P10Error::InvalidArgument << "Tensor must be contiguous for Span2D access";
        }
        return check_dims_for_2d_access<T>();
    }

    template<typename T>
    P10Error check_dims_for_3d_span() const {
        if (!is_contiguous()) {
            return P10Error::InvalidArgument << "Tensor must be contiguous for Span3D access";
        }
        return check_dims_for_3d_access<T>();
    }

    Blob blob_;
    Dtype dtype_;
    Shape shape_;
    Stride stride_;
    Axes axes_;
    bool is_contiguous_ = true;
};
}  // namespace p10
