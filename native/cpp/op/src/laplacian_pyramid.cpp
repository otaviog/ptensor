#include "laplacian_pyramid.hpp"

#include "elemwise.hpp"
#include "resize.hpp"

namespace p10::op {

namespace {
    P10Error validate_process_arguments(const Tensor& input, std::span<Tensor> output);
    P10Error validate_reconstruct_arguments(std::span<const Tensor> pyramid);

}  // namespace

P10Error
LaplacianPyramid::process(const Tensor& in_tensor, std::span<Tensor> out_laplacian_pyr) const {
    if (const auto error = validate_process_arguments(in_tensor, out_laplacian_pyr);
        error.is_error()) {
        return error;
    }

    const auto num_levels = out_laplacian_pyr.size();
    store_gaussian_pyramid(in_tensor, num_levels);
    transform_gaussian_laplacian_pyramid(out_laplacian_pyr);
    return P10Error::Ok;
}

void LaplacianPyramid::store_gaussian_pyramid(const Tensor& in_tensor, size_t num_levels) const {
    gaussian_pyramid_.resize(num_levels);
    gaussian_pyramid_[0] = std::move(in_tensor.clone().unwrap());
    Tensor downsample_buffer;
    for (auto level = 1; level < num_levels; ++level) {
        const auto& prev = gaussian_pyramid_[level - 1];
        const auto half_height = prev.shape(1).unwrap() / 2;
        const auto half_width = prev.shape(2).unwrap() / 2;

        assert(resize(prev, downsample_buffer, half_width, half_height).is_ok());
        blur_op_(downsample_buffer, gaussian_pyramid_[level]);
    }
}

void LaplacianPyramid::transform_gaussian_laplacian_pyramid(std::span<Tensor> output) const {
    const auto num_levels = output.size();
    assert(num_levels == gaussian_pyramid_.size());

    Tensor upsample_buffer;
    for (auto level = 0; level < num_levels - 1; ++level) {
        size_t height = gaussian_pyramid_[level].shape(1).unwrap();
        size_t width = gaussian_pyramid_[level].shape(2).unwrap();

        assert(resize(gaussian_pyramid_[level + 1], upsample_buffer, width, height).is_ok());
        assert(subtract_elemwise(gaussian_pyramid_[level], upsample_buffer, output[level]).is_ok());
    }
    output.back() = std::move(gaussian_pyramid_.back());
}

P10Error LaplacianPyramid::reconstruct(std::span<const Tensor> pyramid, Tensor& output) const {
    if (const auto err = validate_reconstruct_arguments(pyramid); err.is_error()) {
        return err;
    }

    const auto num_levels = static_cast<int>(pyramid.size());
    Tensor upsample_buffer;

    output = std::move(pyramid[num_levels - 1].clone().unwrap());
    for (int level = num_levels - 2; level >= 0; --level) {
        const auto Ll = pyramid[level].clone().unwrap();

        const size_t height = Ll.shape(1).unwrap();
        const size_t width = Ll.shape(2).unwrap();

        assert(resize(output, upsample_buffer, width, height).is_ok());
        assert(add_elemwise(Ll, upsample_buffer, output).is_ok());
    }
    return P10Error::Ok;
}

namespace {
    P10Error validate_process_arguments(const Tensor& input, std::span<Tensor> output) {
        if (input.shape().dims() != 3) {
            return P10Error::InvalidArgument << "Input tensor must be a 3D tensor.";
        }
        if (input.dtype() != Dtype::Float32) {
            return P10Error::InvalidArgument << "Input tensor must be of type FLOAT32.";
        }
        if (output.empty()) {
            return P10Error::InvalidArgument << "Output span is empty.";
        }
        return P10Error::Ok;
    }

    P10Error validate_reconstruct_arguments(std::span<const Tensor> pyramid) {
        if (pyramid.empty()) {
            return P10Error::InvalidArgument << "Input pyramid is empty.";
        }
        for (const auto& level : pyramid) {
            if (level.shape().dims() != 3) {
                return P10Error::InvalidArgument << "All pyramid levels must be 3D tensors.";
            }
            if (level.dtype() != Dtype::Float32) {
                return P10Error::InvalidArgument
                    << "All pyramid levels must be of type FLOAT32.";
            }
        }
        return P10Error::Ok;
    }
}  // namespace
};  // namespace p10::op
