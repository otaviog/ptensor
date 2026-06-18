#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <utility>

#include <type_traits>

#include "accessor3d.hpp"
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
#include "span4d.hpp"
#include "stride.hpp"
#include "tensor_options.hpp"

namespace p10 {

using OptionalDeallocationFunction = std::optional<std::function<void(void*)>>;

/// Controls how an `as_span*`/`as_accessor*` view reconciles the tensor rank
/// with the rank the accessor needs.
enum class RankFit {
    /// Exact rank or error (default). Ops that require a fixed rank get it.
    Strict,
    /// Reach the target rank by adding/removing leading size-1 dims:
    /// `[H,W]` -> `[1,H,W]`, `[1,C,H,W]` -> `[C,H,W]`. A leading dim that is
    /// not size 1 cannot be dropped and yields an error.
    Flexible,
};

/// Tensor class.
///
/// A tensor is a multi-dimensional array of elements of a single data type.
/// The tensor class provides a way to store and manipulate data in a
/// multi-dimensional array.
/// The tensor class can own its data or can be a view of another pointer data.
/// depending if the constructor froblob_view is used with a deallocation function.
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

    P10Error create(
        const Shape& shape,
        const TensorOptions& options = TensorOptions(),
        std::optional<std::reference_wrapper<bool>> new_allocated = std::nullopt
    );

    P10Error create_like(const Tensor& other) {
        return create(other.shape(), other.options());
    }

    /// Clones the tensor if the tensor is in CPU memory
    P10Result<Tensor> clone() const;

    /// Default constructor. Creates an empty tensor.
    Tensor() = default;

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;
    ~Tensor() = default;

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

    /// Gets the usage hint of the tensor (how its contents are meant to be
    /// interpreted; does not affect storage or layout).
    Usage usage() const {
        return usage_;
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

    Tensor as_view() const {
        return Tensor(blob_.view(), shape(), options());
    }

    /// Returns a copy of the tensor options.
    TensorOptions options() const {
        return TensorOptions().dtype(dtype_).stride(stride_).usage(usage_).device(blob_.device());
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

        return Ok(
            Iterator<scalar_t> {
                data_res.unwrap(),
                shape_.as_span(),
                stride_.as_span(),
            }
        );
    }

    template<typename scalar_t>
    P10Result<Iterator<const scalar_t>> iterator() const {
        auto data_res = data_as<scalar_t>();
        if (!data_res.is_ok()) {
            return Err(data_res.error());
        }

        return Ok(
            Iterator<const scalar_t> {
                data_res.unwrap(),
                shape_.as_span(),
                stride_.as_span(),
            }
        );
    }

    template<typename scalar_t>
    P10Result<std::span<const scalar_t>> as_span1d() const {
        auto data_res = data_as<scalar_t>();
        if (!data_res.is_ok()) {
            return Err(data_res.error());
        }
        auto elem_count = size();
        if constexpr (detail::IS_COMPLEX_V<std::remove_const_t<scalar_t>>) {
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
        if constexpr (detail::IS_COMPLEX_V<std::remove_const_t<scalar_t>>) {
            elem_count /= 2;
        }
        return Ok(std::span<scalar_t> {blob_.data<scalar_t>(), elem_count});
    }

    template<typename T, RankFit Fit = RankFit::Strict>
    P10Result<Span2D<T>> as_span2d() {
        if (!is_contiguous()) {
            return Err(P10Error::NotImplemented << "Tensor must be contiguous for Span2D access");
        }
        auto data_res = data_as<T>();
        if (!data_res.is_ok()) {
            return Err(data_res.error());
        }
        auto fit = fit_rank<2, Fit>(logical_rank<T>());
        if (fit.is_error()) {
            return Err(fit.error());
        }
        const auto [shape, stride] = fit.unwrap();
        return Ok(Span2D<T> {data_res.unwrap(), shape[0], shape[1]});
    }

    template<typename T, RankFit Fit = RankFit::Strict>
    P10Result<Span2D<const T>> as_span2d() const {
        if (!is_contiguous()) {
            return Err(P10Error::NotImplemented << "Tensor must be contiguous for Span2D access");
        }
        auto data_res = data_as<const T>();
        if (!data_res.is_ok()) {
            return Err(data_res.error());
        }
        auto fit = fit_rank<2, Fit>(logical_rank<T>());
        if (fit.is_error()) {
            return Err(fit.error());
        }
        const auto [shape, stride] = fit.unwrap();
        return Ok(Span2D<const T> {data_res.unwrap(), shape[0], shape[1]});
    }

    template<typename T, RankFit Fit = RankFit::Strict>
    P10Result<Span3D<T>> as_span3d() {
        if (!is_contiguous()) {
            return Err(P10Error::InvalidArgument << "Tensor must be contiguous for Span3D access");
        }
        auto data_res = data_as<T>();
        if (!data_res.is_ok()) {
            return Err(data_res.error());
        }
        auto fit = fit_rank<3, Fit>(logical_rank<T>());
        if (fit.is_error()) {
            return Err(fit.error());
        }
        const auto [shape, stride] = fit.unwrap();
        return Ok(Span3D<T> {data_res.unwrap(), shape[0], shape[1], shape[2]});
    }

    template<typename T, RankFit Fit = RankFit::Strict>
    P10Result<Span3D<const T>> as_span3d() const {
        if (!is_contiguous()) {
            return Err(P10Error::InvalidArgument << "Tensor must be contiguous for Span3D access");
        }
        auto data_res = data_as<const T>();
        if (data_res.is_error()) {
            return Err(data_res.error());
        }
        auto fit = fit_rank<3, Fit>(logical_rank<T>());
        if (fit.is_error()) {
            return Err(fit.error());
        }
        const auto [shape, stride] = fit.unwrap();
        return Ok(Span3D<const T> {data_res.unwrap(), shape[0], shape[1], shape[2]});
    }

    template<typename T, RankFit Fit = RankFit::Strict>
    P10Result<Span4D<T>> as_span4d() {
        if (!is_contiguous()) {
            return Err(P10Error::InvalidArgument << "Tensor must be contiguous for Span4D access");
        }
        auto data_res = data_as<T>();
        if (!data_res.is_ok()) {
            return Err(data_res.error());
        }
        auto fit = fit_rank<4, Fit>(logical_rank<T>());
        if (fit.is_error()) {
            return Err(fit.error());
        }
        const auto [shape, stride] = fit.unwrap();
        return Ok(Span4D<T> {data_res.unwrap(), shape[0], shape[1], shape[2], shape[3]});
    }

    template<typename T, RankFit Fit = RankFit::Strict>
    P10Result<Span4D<const T>> as_span4d() const {
        if (!is_contiguous()) {
            return Err(P10Error::InvalidArgument << "Tensor must be contiguous for Span4D access");
        }
        auto data_res = data_as<const T>();
        if (data_res.is_error()) {
            return Err(data_res.error());
        }
        auto fit = fit_rank<4, Fit>(logical_rank<T>());
        if (fit.is_error()) {
            return Err(fit.error());
        }
        const auto [shape, stride] = fit.unwrap();
        return Ok(Span4D<const T> {data_res.unwrap(), shape[0], shape[1], shape[2], shape[3]});
    }

    template<typename T, RankFit Fit = RankFit::Strict>
    P10Result<Accessor1D<T>> as_accessor1d() {
        auto data_res = data_as<T>();
        if (!data_res.is_ok()) {
            return Err(data_res.error());
        }
        auto fit = fit_rank<1, Fit>(logical_rank<T>());
        if (fit.is_error()) {
            return Err(fit.error());
        }
        const auto [shape, stride] = fit.unwrap();
        return Ok(Accessor1D<T> {data_res.unwrap(), shape[0], stride[0]});
    }

    template<typename T, RankFit Fit = RankFit::Strict>
    P10Result<Accessor1D<const T>> as_accessor1d() const {
        auto data_res = data_as<const T>();
        if (!data_res.is_ok()) {
            return Err(data_res.error());
        }
        auto fit = fit_rank<1, Fit>(logical_rank<T>());
        if (fit.is_error()) {
            return Err(fit.error());
        }
        const auto [shape, stride] = fit.unwrap();
        return Ok(Accessor1D<const T> {data_res.unwrap(), shape[0], stride[0]});
    }

    template<typename T, RankFit Fit = RankFit::Strict>
    P10Result<Accessor2D<T>> as_accessor2d() {
        auto data_res = data_as<T>();
        if (!data_res.is_ok()) {
            return Err(data_res.error());
        }
        auto fit = fit_rank<2, Fit>(logical_rank<T>());
        if (fit.is_error()) {
            return Err(fit.error());
        }
        const auto [shape, stride] = fit.unwrap();
        return Ok(Accessor2D<T> {data_res.unwrap(), shape, stride});
    }

    template<typename T, RankFit Fit = RankFit::Strict>
    P10Result<Accessor2D<const T>> as_accessor2d() const {
        auto data_res = data_as<const T>();
        if (!data_res.is_ok()) {
            return Err(data_res.error());
        }
        auto fit = fit_rank<2, Fit>(logical_rank<T>());
        if (fit.is_error()) {
            return Err(fit.error());
        }
        const auto [shape, stride] = fit.unwrap();
        return Ok(Accessor2D<const T> {data_res.unwrap(), shape, stride});
    }

    template<typename T, RankFit Fit = RankFit::Strict>
    P10Result<Accessor3D<T>> as_accessor3d() {
        auto data_res = data_as<T>();
        if (!data_res.is_ok()) {
            return Err(data_res.error());
        }
        auto fit = fit_rank<3, Fit>(logical_rank<T>());
        if (fit.is_error()) {
            return Err(fit.error());
        }
        const auto [shape, stride] = fit.unwrap();
        return Ok(Accessor3D<T> {data_res.unwrap(), shape, stride});
    }

    template<typename T, RankFit Fit = RankFit::Strict>
    P10Result<Accessor3D<const T>> as_accessor3d() const {
        auto data_res = data_as<const T>();
        if (!data_res.is_ok()) {
            return Err(data_res.error());
        }
        auto fit = fit_rank<3, Fit>(logical_rank<T>());
        if (fit.is_error()) {
            return Err(fit.error());
        }
        const auto [shape, stride] = fit.unwrap();
        return Ok(Accessor3D<const T> {data_res.unwrap(), shape, stride});
    }

    auto visit(auto&& visitor) {
        assert(is_contiguous());
        return dtype_.visit(std::forward<decltype(visitor)>(visitor), as_bytes());
    }

    auto visit(auto&& visitor) const {
        assert(is_contiguous());
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

    P10Result<Tensor> select_dimension(int64_t dim, int64_t index) const;

    /// Reshapes the tensor to the given shape.
    /// # Arguments
    /// * `new_shape` - The new shape of the tensor.
    /// # Returns
    /// * An error if the reshape is not possible.
    P10Error reshape(const Shape& new_shape);

    /// Returns a view of the tensor with `new_shape`, sharing the same data.
    /// Unlike `reshape`, this does not mutate the tensor. The element count must
    /// match and, for non-contiguous tensors, the layout must allow the reshape.
    P10Result<Tensor> as_reshape(const Shape& new_shape) const;

    /// Returns a 1D view of the tensor with all elements flattened, sharing the
    /// same data. The tensor layout must allow the flatten (e.g. contiguous).
    P10Result<Tensor> ravel() const;

    /// Transposes a 2D tensor.
    /// # Arguments
    /// * `other` - The transposed tensor.
    /// # Returns
    /// * An error if the tensor is not 2D or not contiguous or device is not CPU.
    P10Error transpose(Tensor& other) const;

    P10Error transpose() {
        return transpose(*this);
    }

    P10Error fill(double value);

    P10Error copy_from(const Tensor& src);

    P10Error convert_from(const Tensor& source, const TensorOptions options);

  private:
    Tensor(Blob&& blob, Shape shape, const TensorOptions& options) :
        blob_ {std::move(blob)},
        shape_ {std::move(shape)} {
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
        return Ok(blob_.data<T>());
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

    // Logical rank of the tensor for view purposes: complex tensors carry a
    // trailing size-2 (real/imag) dim that the element type folds away.
    template<typename T>
    size_t logical_rank() const {
        if constexpr (detail::IS_COMPLEX_V<std::remove_const_t<T>>) {
            return dims() - 1;
        } else {
            return dims();
        }
    }

    // Reconciles the tensor's first `d` extents/strides to rank N. RankFit::Strict
    // requires d == N exactly; RankFit::Flexible reaches N by adding (low->high)
    // or dropping leading size-1 dims. `d` is the logical rank (see logical_rank).
    template<size_t N, RankFit Fit>
    P10Result<std::pair<std::array<int64_t, N>, std::array<int64_t, N>>> fit_rank(size_t d) const {
        const auto shape = shape_.as_span();
        const auto stride = stride_.as_span();

        std::array<int64_t, N> out_shape {};
        std::array<int64_t, N> out_stride {};

        if constexpr (Fit == RankFit::Strict) {
            if (d != N) {
                return Err(
                    P10Error::InvalidArgument
                    << "Tensor rank does not match the requested view rank"
                );
            }
        }

        if (d == N) {
            for (size_t i = 0; i < N; ++i) {
                out_shape[i] = shape[i];
                out_stride[i] = stride[i];
            }
        } else if (d < N) {
            const size_t pad = N - d;
            const int64_t lead = d > 0 ? shape[0] * stride[0] : 1;
            for (size_t i = 0; i < pad; ++i) {
                out_shape[i] = 1;
                out_stride[i] = lead;
            }
            for (size_t i = 0; i < d; ++i) {
                out_shape[pad + i] = shape[i];
                out_stride[pad + i] = stride[i];
            }
        } else {
            const size_t drop = d - N;
            for (size_t i = 0; i < drop; ++i) {
                if (shape[i] != 1) {
                    return Err(
                        P10Error::InvalidArgument,
                        "Cannot fit rank: leading dimension is not size 1"
                    );
                }
            }
            for (size_t i = 0; i < N; ++i) {
                out_shape[i] = shape[drop + i];
                out_stride[i] = stride[drop + i];
            }
        }
        return Ok(std::make_pair(out_shape, out_stride));
    }

    Blob blob_;
    Dtype dtype_;
    Shape shape_;
    Stride stride_;
    Usage usage_ = Usage::NotSpecified;
    bool is_contiguous_ = true;
};
}  // namespace p10
