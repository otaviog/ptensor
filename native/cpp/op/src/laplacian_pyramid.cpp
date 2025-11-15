#include "laplacian_pyramid.hpp"

#include "elemwise.hpp"
#include "ptensor/p10_error.hpp"
#include "resize.hpp"

namespace p10::op {

namespace {
    P10Error validate_process_arguments(const Tensor& input, std::span<Tensor> output);
    P10Error validate_reconstruct_arguments(std::span<const Tensor> pyramid);

}  // namespace

P10Error
LaplacianPyramid::transform(const Tensor& in_tensor, std::span<Tensor> out_laplacian_pyr) const {
    if (const auto error = validate_process_arguments(in_tensor, out_laplacian_pyr);
        error.is_error()) {
        return error;
    }

    const auto num_levels = out_laplacian_pyr.size();
    store_gaussian_pyramid(in_tensor, num_levels);
    pyramid_from_gaussian_to_laplacian(out_laplacian_pyr);
    return P10Error::Ok;
}

void LaplacianPyramid::store_gaussian_pyramid(const Tensor& in_tensor, size_t num_levels) const {
    gaussian_pyramid_.resize(num_levels);
    gaussian_pyramid_[0] = in_tensor.clone().unwrap();
    Tensor blur_buffer;
    for (size_t level = 1; level < num_levels; ++level) {
        const auto& prev = gaussian_pyramid_[level - 1];
        const auto half_height = prev.shape(1).unwrap() / 2;
        const auto half_width = prev.shape(2).unwrap() / 2;

        blur_op_.transform(prev, blur_buffer).expect("Blur failed");
        resize(blur_buffer, gaussian_pyramid_[level], half_width, half_height).expect("Resize failed");
    }
}

void LaplacianPyramid::pyramid_from_gaussian_to_laplacian(std::span<Tensor> output) const {
    const auto num_levels = output.size();
    assert(num_levels == gaussian_pyramid_.size());

    Tensor upsample_buffer;
    for (size_t level = 0; level < num_levels - 1; ++level) {
        size_t height = gaussian_pyramid_[level].shape(1).unwrap();
        size_t width = gaussian_pyramid_[level].shape(2).unwrap();

        resize(gaussian_pyramid_[level + 1], upsample_buffer, width, height)
            .expect("Upsample failed");
        subtract_elemwise(gaussian_pyramid_[level], upsample_buffer, output[level])
            .expect("Subtract failed");
    }
    output.back() = gaussian_pyramid_.back().clone().expect("Clone failed");
}
 
P10Error LaplacianPyramid::reconstruct(std::span<const Tensor> pyramid, Tensor& output) const {
    if (const auto err = validate_reconstruct_arguments(pyramid); err.is_error()) {
        return err;
    }

    const auto num_levels = static_cast<int>(pyramid.size());
    Tensor upsample_buffer;

    // Allocate output tensor with the shape of the largest level
    // so we can avoid reallocations during the loop
    // and don't change the output if preallocated
    P10_RETURN_IF_ERROR(output.create(pyramid[0].shape(), pyramid[0].dtype()));
    P10_RETURN_IF_ERROR(output.copy_from(pyramid.back()));
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
                return P10Error::InvalidArgument << "All pyramid levels must be of type FLOAT32.";
            }
        }
        return P10Error::Ok;
    }
}  // namespace
};  // namespace p10::op
