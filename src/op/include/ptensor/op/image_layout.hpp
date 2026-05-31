#pragma once

#include <optional>

#include <ptensor/dtype.hpp>
#include <ptensor/p10_error.hpp>

namespace p10 {
class Tensor;
}

namespace p10::op {

/// Options for `image_to_tensor`.
class ImageToTensorOptions {
  public:
    /// Target dtype of the output tensor. If unset, defaults to `Dtype::Float32`.
    std::optional<Dtype> target_dtype() const {
        return target_type_;
    }

    ImageToTensorOptions& target_dtype(std::optional<Dtype> dtype) {
        target_type_ = dtype;
        return *this;
    }

    /// When converting between uint8 and a floating-point dtype, rescale values
    /// into the target range:
    ///
    /// * uint8 -> float: `x / 255`
    /// * float -> uint8: `round(x * 255)`
    ///
    /// Has no effect when source and target dtype are both integer.
    bool normalize() const {
        return normalize_;
    }

    ImageToTensorOptions& normalize(bool normalize) {
        normalize_ = normalize;
        return *this;
    }

    /// If true, prepend a batch dimension of size 1 to the output shape:
    /// `[C, H, W]` becomes `[1, C, H, W]`.
    bool unsqueeze() const {
        return unsqueeze_;
    }

    ImageToTensorOptions& unsqueeze(bool unsqueeze) {
        unsqueeze_ = unsqueeze;
        return *this;
    }

  private:
    std::optional<Dtype> target_type_ = std::nullopt;
    bool normalize_ = false;
    bool unsqueeze_ = false;
};

/// Convert an image tensor `[H, W, C]` (uint8) to a planar tensor `[C, H, W]`,
/// or `[1, C, H, W]` when `options.unsqueeze()` is true.
///
/// # Arguments
///
/// * `image` - Input image tensor. Must be uint8, 3D `[height, width, channels]`,
///             and contiguous in memory.
/// * `out_tensor` - Output planar tensor. Overwritten on success.
/// * `options` - Conversion options. See `ImageToTensorOptions`.
///
/// # Returns
///
/// * `P10Error::Ok` on success.
/// * `P10Error::InvalidArgument` if `image` is not uint8, not 3D, or not contiguous.
P10Error image_to_tensor(
    const Tensor& image,
    Tensor& out_tensor,
    const ImageToTensorOptions& options = ImageToTensorOptions()
);

/// Convert a planar floating-point tensor `[C, H, W]` or `[1, C, H, W]` to an
/// image tensor `[H, W, C]`.
///
/// When the target dtype is integer, values are scaled by 255 and clamped to
/// `[0, 255]`. When the target dtype is floating, values are copied as-is.
///
/// # Arguments
///
/// * `tensor` - Input planar tensor. Must be float32 or float64, contiguous,
///              and have shape `[C, H, W]` or `[1, C, H, W]`.
/// * `out_image_tensor` - Output image tensor. Overwritten on success.
/// * `target_dtype` - Output dtype. Defaults to `Dtype::Uint8` if unset.
///
/// # Returns
///
/// * `P10Error::Ok` on success.
/// * `P10Error::InvalidArgument` if `tensor` is not floating-point, not 3D/4D,
///   not contiguous, or has a 4D shape whose leading dim is not 1.
P10Error image_from_tensor(
    const Tensor& tensor,
    Tensor& out_image_tensor,
    std::optional<Dtype> target_dtype = std::nullopt
);

}  // namespace p10::op
