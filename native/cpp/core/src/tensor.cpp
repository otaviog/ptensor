#include "tensor.hpp"

#include <algorithm>
#include <cassert>

#include "ptensor_error.hpp"

namespace p10 {
namespace {
    PtensorError are_options_valid_for_creation(const TensorOptions& options);
    size_t compute_size_bytes(const Shape& shape, const Dtype& dtype);
}  // namespace

PtensorResult<Tensor> Tensor::full(const Shape& shape, double value, const TensorOptions& options) {
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

PtensorResult<Tensor> Tensor::empty(const Shape& shape, const TensorOptions& options) {
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

PtensorError Tensor::create(const Shape& shape, const TensorOptions& options) {
    if (auto status = are_options_valid_for_creation(options); !status.is_ok()) {
        return status;
    }

    const auto ask_size = compute_size_bytes(shape, options.dtype());
    if (ask_size <= size_bytes()) {
        shape_ = shape;
        set_options(options);
        return PtensorError::Ok;
    }

    blob_ = Blob::allocate(ask_size);
    return PtensorError::Ok;
}

PtensorResult<Tensor> Tensor::clone() const {
    if (blob_.device() != Device::Cpu) {
        return Err(PtensorError::NotImplemented);
    }

    if (!is_contiguous()) {
        return Err(PtensorError::NotImplemented);
    }

    const auto blob_size = size_bytes();

    auto new_blob = Blob::allocate(blob_size);
    std::memcpy(new_blob.data<uint8_t>(), blob_.data<uint8_t>(), blob_size);

    return Ok(Tensor(std::move(new_blob), shape_, options()));
}

bool Tensor::is_contiguous() const {
    for (int i = ((int)shape_.dims()) - 2; i >= 0; --i) {
        if (stride_[i].unwrap() != shape_[i + 1].unwrap() * stride_[i + 1].unwrap()) {
            return false;
        }
    }
    return true;
}

PtensorResult<Tensor> Tensor::to_contiguous() const {
    if (is_contiguous()) {
        return clone();
    }

    if (blob_.device() != Device::Cpu) {
        return Err(PtensorError::NotImplemented);
    }

    Tensor contiguous_tensor = empty(shape_, options().clone().stride(Stride())).unwrap();
    assert(contiguous_tensor.is_contiguous());

    dtype_.visit(
        [this](auto dest_span) {
            using scalar_t =
                decltype(dest_span)::element_type;  // std::remove_pointer_t<decltype(dest_ptr)>;
            const size_t num_elements = this->size();
            const int ndims = this->shape_.dims();

            std::array<int64_t, P10_MAX_SHAPE> coords {};
            const auto source_span = this->as_span1d<scalar_t>();

            const auto this_stride = this->stride_.as_span();
            for (size_t i = 0; i < num_elements; i++) {
                int64_t from_index = 0;

                for (int j = 0; j < ndims; j++) {
                    from_index += coords[j] * this_stride[j];
                }

                dest_span[i] = source_span[from_index];
                for (auto j = ndims - 1; j >= 0; j--) {
                    if (coords[j] < shape(j).unwrap() - 1) {
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

namespace {
    PtensorError are_options_valid_for_creation(const TensorOptions& options) {
        if (options.device() != Device::Cpu) {
            return PtensorError::InvalidArgument
                << "Cannot create tensor outside of the CPU, allocate using your device API";
        }

        return PtensorError::Ok;
    }

    size_t compute_size_bytes(const Shape& shape, const Dtype& dtype) {
        return shape.count() * dtype.size();
    }
}  // namespace

}  // namespace p10
