#pragma once

#include <vector>

#include <ptensor/p10_result.hpp>
#include <ptensor/tensor.hpp>

#include "blur.hpp"

namespace p10::op {
/// Laplacian pyramid decomposition and reconstruction.
///
/// Decomposes an input tensor into a multi-scale Laplacian pyramid, representing
/// difference-of-Gaussians at different scales. This is useful for image analysis,
/// blending, and preprocessing tasks.
class LaplacianPyramid {
  public:
    /// Creates a Laplacian pyramid operator.
    ///
    /// # Arguments
    ///
    /// * `blur_kernel_size`: Size of the Gaussian blur kernel (must be odd)
    /// * `blur_sigma`: Standard deviation of the Gaussian kernel
    ///
    /// # Returns
    ///
    /// A new LaplacianPyramid operator or an error if the blur operator
    /// creation fails.
    static P10Result<LaplacianPyramid>
    create(size_t blur_kernel_size = 5, float blur_sigma = 1.0f) {
        auto blur_op_result = GaussianBlur::create(blur_kernel_size, blur_sigma);
        if (blur_op_result.is_error()) {
            return Err(blur_op_result.unwrap_err());
        }
        return Ok(LaplacianPyramid(blur_op_result.unwrap()));
    }

    /// Decomposes an input tensor into a Laplacian pyramid.
    ///
    /// Generates a multi-scale pyramid representation where each level contains
    /// the difference between the current Gaussian-blurred level and the next
    /// coarser level.
    ///
    /// # Arguments
    ///
    /// * `input`: The input tensor to decompose
    /// * `output`: Span to store the pyramid levels (must be pre-allocated)
    ///
    /// # Returns
    ///
    /// An error if decomposition fails, otherwise success.
    P10Error transform(const Tensor& input, std::span<Tensor> output);

    /// Reconstructs a tensor from a Laplacian pyramid.
    ///
    /// Collapses a multi-scale Laplacian pyramid back into a single tensor by
    /// progressively upsampling and adding successive levels.
    ///
    /// # Arguments
    ///
    /// * `pyramid`: Span of pyramid levels to reconstruct from
    /// * `output`: The reconstructed tensor
    ///
    /// # Returns
    ///
    /// An error if reconstruction fails, otherwise success.
    P10Error reconstruct(std::span<const Tensor> pyramid, Tensor& output) const;

  private:
    explicit LaplacianPyramid(const GaussianBlur& blur_op) : blur_op_(blur_op) {}

    void store_gaussian_pyramid(const Tensor& input, size_t num_levels);
    void pyramid_from_gaussian_to_laplacian(std::span<Tensor> output) const;

    GaussianBlur blur_op_;
    std::vector<Tensor> gaussian_pyramid_;
};
}  // namespace p10::op
