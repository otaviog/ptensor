#pragma once

#include <vector>

#include <ptensor/p10_result.hpp>
#include <ptensor/tensor.hpp>

#include "blur.hpp"

namespace p10::op {
class LaplacianPyramid {
  public:
    static P10Result<LaplacianPyramid>
    create(size_t blur_kernel_size = 5, float blur_sigma = 1.0f) {
        auto blur_op_result = GaussianBlur::create(blur_kernel_size, blur_sigma);
        if (blur_op_result.is_error()) {
            return Err(blur_op_result.unwrap_err());
        }
        return Ok(LaplacianPyramid(blur_op_result.unwrap()));
    }

    P10Error transform(const Tensor& input, std::span<Tensor> output) const;

    P10Error reconstruct(std::span<const Tensor> pyramid, Tensor& output) const;

  private:
    explicit LaplacianPyramid(const GaussianBlur& blur_op) : blur_op_(blur_op) {}

    void store_gaussian_pyramid(const Tensor& input, size_t num_levels) const;
    void pyramid_from_gaussian_to_laplacian(std::span<Tensor> output) const;

    GaussianBlur blur_op_;
    mutable std::vector<Tensor> gaussian_pyramid_;
};
};  // namespace p10::op
