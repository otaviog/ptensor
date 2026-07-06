#pragma once

#include <optional>

#include <ptensor/dtype.hpp>
#include <ptensor/p10_error.hpp>

namespace p10 {
class Tensor;
}

namespace p10::op {

class ImageFromTensorOptions;

/// Options for `image_to_tensor`.
class ImageToTensorOptions {
  public:
    ImageToTensorOptions() = default;

    /// Mirror the shared settings of a from-tensor conversion so the two
    /// directions stay symmetric. Copies `normalize`; the output dtype keeps
    /// this direction's default (float32) unless set explicitly.
    explicit ImageToTensorOptions(const ImageFromTensorOptions& other);

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

/// Options for `image_from_tensor`. Mirrors the shared knobs of
/// `ImageToTensorOptions` with matching defaults, so a default-to-default round
/// trip preserves values without any configuration.
class ImageFromTensorOptions {
  public:
    ImageFromTensorOptions() = default;

    /// Mirror the shared settings of a to-tensor conversion so a round trip
    /// stays symmetric. Copies `normalize`; the output dtype keeps this
    /// direction's default (uint8) unless set explicitly.
    explicit ImageFromTensorOptions(const ImageToTensorOptions& other) :
        normalize_(other.normalize()) {}

    /// Target dtype of the output image tensor. If unset, defaults to `Dtype::Uint8`.
    std::optional<Dtype> target_dtype() const {
        return target_type_;
    }

    ImageFromTensorOptions& target_dtype(std::optional<Dtype> dtype) {
        target_type_ = dtype;
        return *this;
    }

    /// When the target dtype is integer, rescale `float -> uint8` as
    /// `round(x * 255)` (values then clamped to `[0, 255]`). When false, values
    /// are cast as-is and clamped. Has no effect for a floating-point target.
    bool normalize() const {
        return normalize_;
    }

    ImageFromTensorOptions& normalize(bool normalize) {
        normalize_ = normalize;
        return *this;
    }

  private:
    std::optional<Dtype> target_type_ = std::nullopt;
    bool normalize_ = false;
};

inline ImageToTensorOptions::ImageToTensorOptions(const ImageFromTensorOptions& other) :
    normalize_(other.normalize()) {}

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
/// image tensor `[H, W, C]`. This is the inverse of `image_to_tensor`; build the
/// options from the ones used there (`ImageFromTensorOptions(to_options)`) to
/// keep a round trip symmetric.
///
/// With `options.normalize()` and an integer target, values are scaled by 255
/// and clamped to `[0, 255]`; otherwise values are cast as-is and clamped. For a
/// floating-point target, values are copied unchanged.
///
/// # Arguments
///
/// * `tensor` - Input planar tensor. Must be float32 or float64, contiguous,
///              and have shape `[C, H, W]` or `[1, C, H, W]`.
/// * `out_image_tensor` - Output image tensor. Overwritten on success.
/// * `options` - Conversion options. See `ImageFromTensorOptions`.
///
/// # Returns
///
/// * `P10Error::Ok` on success.
/// * `P10Error::InvalidArgument` if `tensor` is not floating-point, not 3D/4D,
///   not contiguous, or has a 4D shape whose leading dim is not 1.
P10Error image_from_tensor(
    const Tensor& tensor,
    Tensor& out_image_tensor,
    const ImageFromTensorOptions& options = ImageFromTensorOptions()
);

}  // namespace p10::op
