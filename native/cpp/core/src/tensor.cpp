#include "tensor.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>

#include <type_traits>

#include "p10_error.hpp"
#include "ptensor/config.h"
#include "shape.hpp"
#include "stride.hpp"

namespace p10 {
namespace {
    P10Error are_options_valid_for_creation(const TensorOptions& options);
    size_t compute_size_bytes(const Shape& shape, const Dtype& dtype);
    bool is_stride_contiguous(const Stride& stride, const Shape& shape);
    template<typename Iter1, typename Iter2>
    void copy_one_except(Iter1 begin, Iter1 end, size_t index, Iter2 out);
    P10Error
    check_reshapeability(const Shape& old_shape, const Shape& new_shape, bool is_contiguous);

}  // namespace

P10Result<Tensor> Tensor::full(const Shape& shape, double value, const TensorOptions& options) {
    if (shape.count() < 1) {
        return Ok(Tensor(options));
    }

    const auto size = shape.count() * options.dtype().size_bytes();
    auto blob = Blob::allocate(size);
    options.dtype().visit(
        [value, &shape](auto span) {
            using scalar_t = decltype(span)::element_type;
            std::fill(span.data(), span.data() + shape.count(), static_cast<scalar_t>(value));
        },
        std::span(blob.data<std::byte>(), size)
    );

    return Ok(Tensor(std::move(blob), shape, options));
}

P10Result<Tensor> Tensor::from_range(const Shape& shape, const TensorOptions& options, int64_t start) {
    auto result_res = Tensor::zeros(shape, options);
    if (result_res.is_error()) {
        return Err(result_res);
    }

    auto total_size =
        std::accumulate(shape.begin(), shape.end(), int64_t(1), std::multiplies<int64_t>());

    auto result = result_res.unwrap();
    options.dtype().visit(
        [&](auto span) {
            using SpanType = decltype(span)::value_type;
            for (auto i = 0; i < total_size; i++) {
                span[i] = SpanType(start + i);
            }
        },
        result.as_bytes()
    );
    return Ok(std::move(result));
}

P10Result<Tensor> Tensor::empty(const Shape& shape, const TensorOptions& options) {
    if (auto status = are_options_valid_for_creation(options); !status.is_ok()) {
        return Err(status);
    }

    const auto size = compute_size_bytes(shape, options.dtype());
    if (size == 0) {
        return Ok(Tensor(Blob(), shape, options));
    }

    auto blob = Blob::allocate(size);
    return Ok(Tensor(std::move(blob), shape, options));
}

Tensor::Tensor(Tensor&& other) :
    blob_(std::move(other.blob_)),
    dtype_(other.dtype_),
    shape_(std::move(other.shape_)),
    stride_(std::move(other.stride_)),
    axes_(std::move(other.axes_)),
    is_contiguous_(other.is_contiguous_) {
    other.dtype_ = Dtype::Float32;
}

Tensor& Tensor::operator=(Tensor&& other) {
    blob_ = std::move(other.blob_);
    dtype_ = other.dtype_;
    shape_ = std::move(other.shape_);
    stride_ = std::move(other.stride_);
    axes_ = std::move(other.axes_);
    is_contiguous_ = other.is_contiguous_;
    other.dtype_ = Dtype::Float32;
    return *this;
}

P10Error Tensor::create(const Shape& shape, const TensorOptions& options) {
    if (auto status = are_options_valid_for_creation(options); !status.is_ok()) {
        return status;
    }

    const auto ask_size = compute_size_bytes(shape, options.dtype());
    if (ask_size > size_bytes()) {
        blob_ = Blob::allocate(ask_size);
    }

    shape_ = shape;
    set_options(options);

    return P10Error::Ok;
}

P10Result<Tensor> Tensor::clone() const {
    if (blob_.device() != Device::Cpu) {
        return Err(P10Error::NotImplemented);
    }

    if (!is_contiguous()) {
        return Err(P10Error::NotImplemented);
    }

    const auto blob_size = size_bytes();

    auto new_blob = Blob::allocate(blob_size);
    std::memcpy(new_blob.data<uint8_t>(), blob_.data<uint8_t>(), blob_size);

    return Ok(Tensor(std::move(new_blob), shape_, options()));
}

P10Result<Tensor> Tensor::to_contiguous() const {
    if (is_contiguous()) {
        return clone();
    }

    if (blob_.device() != Device::Cpu) {
        return Err(P10Error::NotImplemented);
    }

    Tensor contiguous_tensor = empty(shape_, options().clone().stride(Stride())).unwrap();
    assert(contiguous_tensor.is_contiguous());

    dtype_.visit(
        [this](auto dest_span) {
            using scalar_t =
                decltype(dest_span)::element_type;  // std::remove_pointer_t<decltype(dest_ptr)>;
            const size_t num_elements = this->size();
            const auto shape = this->shape_.as_span();
            const size_t ndims = this->shape_.dims();

            std::array<int64_t, P10_MAX_SHAPE> coords {};
            const auto source_span = this->as_span1d<scalar_t>().unwrap();

            const auto this_stride = this->stride_.as_span();
            for (size_t i = 0; i < num_elements; i++) {
                int64_t from_index = 0;

                for (size_t j = 0; j < ndims; j++) {
                    from_index += coords[j] * this_stride[j];
                }

                dest_span[i] = source_span[from_index];
                for (int j = int(ndims) - 1; j >= 0; j--) {
                    if (coords[j] < shape[j] - 1) {
                        coords[j] += 1;
                        break;
                    } else {
                        coords[j] = 0;
                    }
                }
            }
        },
        contiguous_tensor.as_bytes()
    );

    return Ok(std::move(contiguous_tensor));
}

void Tensor::squeeze() {
    std::span<int64_t> shape = shape_.as_span();
    auto one_pos = shape.begin();
    while (shape.size() > 1) {
        one_pos = std::find(one_pos, shape.end(), 1);
        if (one_pos == shape.end()) {
            break;
        }

        for (auto b = one_pos; b != shape.end() - 1; ++b) {
            *b = *(b + 1);
        }
        shape = shape.subspan(0, shape.size() - 1);
    }
    // TODO: fix strides
    shape_ = make_shape(shape).unwrap();
}

P10Error Tensor::unsqueeze(int64_t dim) {
    if (dim < 0 || dim > int64_t(dims())) {
        return P10Error::InvalidArgument << "Cannot unsqueeze at dimension " + std::to_string(dim)
            + ": must be in range [0, " + std::to_string(dims()) + "]";
    }

    auto new_shape_res = Shape::zeros(size_t(dims() + 1));
    if (new_shape_res.is_error()) {
        return new_shape_res.err();
    }

    auto new_stride_res = Stride::zeros(size_t(dims() + 1));
    if (new_stride_res.is_error()) {
        return new_stride_res.err();
    }

    auto old_shape_s = shape_.as_span();

    auto new_shape = new_shape_res.unwrap();
    auto new_shape_s = new_shape.as_span();

    auto stride_s = stride_.as_span();
    auto new_stride = new_stride_res.unwrap();
    auto new_stride_s = new_stride.as_span();

    if (dim == 0) {
        // When inserting at position 0, stride should be the product of all dimensions in the old shape
        int64_t prod = 1;
        for (size_t i = 0; i < old_shape_s.size(); ++i) {
            prod *= old_shape_s[i];
        }
        new_stride_s[0] = prod;
    } else {
        new_stride_s[0] = stride_s[0];
    }
    for (size_t i = 1; i <= size_t(dim); ++i) {
        new_stride_s[i] = stride_s[i - 1];
    }
    for (size_t i = size_t(dim) + 1; i < new_stride.dims(); ++i) {
        new_stride_s[i] = stride_s[i - 1];
    }
    for (size_t i = 0; i < size_t(dim); ++i) {
        new_shape_s[i] = old_shape_s[i];
    }
    new_shape_s[size_t(dim)] = 1;
    for (size_t i = size_t(dim); i < dims(); ++i) {
        new_shape_s[i + 1] = old_shape_s[i];
    }

    shape_ = new_shape;
    stride_ = new_stride;
    return P10Error::Ok;
}

P10Result<Tensor> Tensor::select_dimension(int64_t dim, int64_t index) {
    if (dim >= int64_t(dims()) || dim < 0) {
        return Err(
            P10Error::InvalidArgument << "Cannot select dimension " + std::to_string(dim)
                + ": must be in range [0, " + std::to_string(dims()) + ")"
        );
    }

    if (index < 0 || index >= shape_[dim].unwrap()) {
        return Err(
            P10Error::InvalidArgument << "Cannot select index " + std::to_string(index)
                + " from dimension " + std::to_string(dim) + ": must be in range [0, "
                + std::to_string(shape_[dim].unwrap()) + ")"
        );
    }

    Shape select_shape(size_t(dims() - 1));
    copy_one_except(shape_.begin(), shape_.end(), size_t(dim), select_shape.begin());
    Stride select_stride = Stride::zeros(size_t(dims() - 1)).unwrap();
    copy_one_except(stride_.begin(), stride_.end(), size_t(dim), select_stride.begin());

    const auto offset = stride_[dim].unwrap() * index * dtype_.size_bytes();

    Tensor select(blob_.view(offset), select_shape, options().stride(select_stride));
    select.is_contiguous_ = (dim == 0) ? is_contiguous_ : false;
    return Ok(std::move(select));
}

void Tensor::set_options(const TensorOptions& options) {
    dtype_ = options.dtype();
    if (options.stride().empty()) {
        stride_ = Stride::from_contiguous_shape(shape_);
        is_contiguous_ = true;
    } else {
        stride_ = options.stride();
        is_contiguous_ = is_stride_contiguous(stride_, shape_);
    }

    axes_ = options.axes().empty() ? Axes(shape_.dims()) : options.axes();
}

P10Error Tensor::reshape(const Shape& new_shape) {
    Stride new_stride = Stride::from_contiguous_shape(new_shape);

    P10_RETURN_IF_ERROR(check_reshapeability(shape(), new_shape, is_contiguous()));

    shape_ = new_shape;
    stride_ = new_stride;
    is_contiguous_ = is_stride_contiguous(new_stride, new_shape);

    return P10Error::Ok;
}

P10Error Tensor::transpose(Tensor& other) const {
    if (dims() != 2) {
        return P10Error::InvalidArgument << "Tensor must have 2 dimensions for transpose";
    }

    if (blob_.device() != Device::Cpu) {
        return P10Error::NotImplemented << "Transpose is only implemented for CPU tensors";
    }

    if (!is_contiguous()) {
        return P10Error::NotImplemented << "Transpose is only implemented for contiguous tensors";
    }

    Shape trans_shape = make_shape(shape_[1].unwrap(), shape_[0].unwrap());
    other.create(trans_shape, dtype());
    dtype_.visit(
        [this, &other](auto type_span) {
            using scalar_t = std::remove_const_t<typename decltype(type_span)::element_type>;
            auto src_span = this->as_span2d<const scalar_t>().unwrap();
            auto dest_span = other.as_span2d<scalar_t>().unwrap();
            const auto rows = src_span.height();
            const auto cols = src_span.width();

            for (size_t r = 0; r < rows; r++) {
                const auto src_row = src_span.row(r);
                for (size_t c = 0; c < cols; c++) {
                    dest_span.row(c)[r] = src_row[c];
                }
            }
        },
        as_bytes()
    );
    return P10Error::Ok;
}

P10Error Tensor::fill(double value) {
    if (device() != Device::Cpu) {
        return P10Error::NotImplemented << "Fill is only implemented for CPU tensors";
    }

    visit([value](auto span) {
        using scalar_t = typename std::decay_t<decltype(span)>::value_type;
        std::fill(span.begin(), span.end(), static_cast<scalar_t>(value));
    });
    return P10Error::Ok;
}

P10Error Tensor::copy_from(const Tensor& src) {
    if (!src.is_contiguous()) {
        return P10Error::NotImplemented << "Copy is only implemented for contiguous tensors";
    }

    P10_RETURN_IF_ERROR(create(src.shape(), src.options()));

    std::memcpy(as_bytes().data(), src.as_bytes().data(), src.size_bytes());

    return P10Error::Ok;
}

namespace {
    P10Error are_options_valid_for_creation(const TensorOptions& options) {
        if (options.device() != Device::Cpu) {
            return P10Error::InvalidArgument
                << "Cannot create tensor outside of the CPU, allocate using your device API";
        }

        return P10Error::Ok;
    }

    size_t compute_size_bytes(const Shape& shape, const Dtype& dtype) {
        return shape.count() * dtype.size_bytes();
    }

    bool is_stride_contiguous(const Stride& stride, const Shape& shape) {
        if (shape.dims() < 2) {
            return true;
        }
        for (int i = ((int)shape.dims()) - 2; i >= 0; --i) {
            if (stride[i].unwrap() != shape[i + 1].unwrap() * stride[i + 1].unwrap()) {
                return false;
            }
        }
        return true;
    }

    template<typename Iter1, typename Iter2>
    void copy_one_except(Iter1 begin, Iter1 end, size_t index, Iter2 out) {
        std::copy(begin, begin + index, out);
        std::copy(begin + index + 1, end, out + index);
    }

    P10Error
    check_reshapeability(const Shape& old_shape, const Shape& new_shape, bool is_contiguous) {
        if (old_shape.count() != new_shape.count()) {
            return P10Error::InvalidArgument << "Cannot reshape tensor of size "
                + std::to_string(old_shape.count()) + " to shape with size "
                + std::to_string(new_shape.count());
        }

        const auto dims = old_shape.dims();

        if (!is_contiguous && old_shape.count() > 0 && new_shape.count() > 0) {
            // Check if the reshape is possible with the current stride
            size_t old_dim = 0;
            size_t new_dim = 0;
            size_t old_size = old_shape[0].unwrap();
            size_t new_size = new_shape[0].unwrap();

            while (old_dim < dims && new_dim < new_shape.dims()) {
                if (old_size == new_size) {
                    old_dim++;
                    new_dim++;
                    if (old_dim < dims) {
                        old_size = old_shape[old_dim].unwrap();
                    }
                    if (new_dim < new_shape.dims()) {
                        new_size = new_shape[new_dim].unwrap();
                    }
                } else if (old_size < new_size) {
                    old_dim++;
                    if (old_dim < dims) {
                        old_size *= old_shape[old_dim].unwrap();
                    }
                } else {  // old_size > new_size
                    new_dim++;
                    if (new_dim < new_shape.dims()) {
                        new_size *= new_shape[new_dim].unwrap();
                    }
                }
            }

            if (old_dim != dims || new_dim != new_shape.dims()) {
                return P10Error::InvalidArgument
                    << "Cannot reshape tensor with non-contiguous layout to the desired shape";
            }
        }

        return P10Error::Ok;
    }

}  // namespace

}  // namespace p10
