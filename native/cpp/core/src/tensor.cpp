#include "tensor.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>

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

}  // namespace

P10Result<Tensor> Tensor::full(const Shape& shape, double value, const TensorOptions& options) {
    if (shape.count() < 1) {
        return Ok(Tensor(options));
    }

    const auto size = shape.count() * options.dtype().size();
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

P10Result<Tensor> Tensor::from_range(const Shape& shape, const Dtype& dtype, int64_t start) {
    auto result_res = Tensor::zeros(shape, dtype);
    if (auto status = result_res.is_error()) {
        return Err(result_res);
    }

    auto total_size =
        std::accumulate(shape.begin(), shape.end(), int64_t(1), std::multiplies<int64_t>());

    auto result = result_res.unwrap();
    dtype.visit(
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
    if (shape.count() < 1) {
        return Ok(Tensor(options));
    }

    if (auto status = are_options_valid_for_creation(options); !status.is_ok()) {
        return Err(status);
    }

    const auto size = compute_size_bytes(shape, options.dtype());
    auto blob = Blob::allocate(size);
    return Ok(Tensor(std::move(blob), shape, options));
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

    shape_ = make_shape(shape).unwrap();
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
    Stride select_stride(size_t(dims() - 1));
    copy_one_except(stride_.begin(), stride_.end(), size_t(dim), select_stride.begin());

    const auto offset = stride_[dim].unwrap() * index * dtype_.size();

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

namespace {
    P10Error are_options_valid_for_creation(const TensorOptions& options) {
        if (options.device() != Device::Cpu) {
            return P10Error::InvalidArgument
                << "Cannot create tensor outside of the CPU, allocate using your device API";
        }

        return P10Error::Ok;
    }

    size_t compute_size_bytes(const Shape& shape, const Dtype& dtype) {
        return shape.count() * dtype.size();
    }

    bool is_stride_contiguous(const Stride& stride, const Shape& shape) {
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

}  // namespace

}  // namespace p10
