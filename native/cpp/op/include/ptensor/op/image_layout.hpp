#pragma once

#include <optional>

#include <ptensor/dtype.hpp>
#include <ptensor/p10_error.hpp>

namespace p10 {
class Tensor;
}

namespace p10::op {

enum class ImageToTensorSqueeze { Squeeze, Unsqueze };

enum class ImageToTensorNormalize { Normalize, KeepValues };

/// Convert an image tensor to a float tensor suitable for model input.
/// The input image tensor must have dtype UINT8 and shape [height, width, channels].
/// The output float tensor will have shape [channels, height, width] and dtype keep unchanged or converted to target_dtype if specified.
///
/// # Arguments
///
/// * `image` - The input image tensor to convert. Must have shape [height, width, channels].
/// * `out_tensor` - The output tensor to store the resulting tensor with shape [channels, height, width] and dtype target_dtype if specified.
/// * `target_dtype` - Optional target dtype for the output tensor. If not specified, the output dtype will be the same as the input.
/// * `normalize_opt` - Whether to normalize pixel values to [0, 1] by dividing by 255. Only applies to float output dtypes. Default is KeepValues.
/// * `squeeze_opt` - Whether to squeeze the output tensor to remove dimensions of size 1. Default is Squeeze.
///
/// # Returns
///
///
P10Error image_to_tensor(
    const Tensor& image,
    Tensor& out_tensor,
    std::optional<Dtype> target_dtype = std::nullopt,
    ImageToTensorNormalize normalize_opt = ImageToTensorNormalize::KeepValues,
    ImageToTensorSqueeze squeeze_opt = ImageToTensorSqueeze::Squeeze
);

/// Convert a tensor back to an image tensor.
/// The input tensor must have shape [channels, height, width].
/// The output image tensor will have shape [height, width, channels], and converted to target_dtype if specified.
///
/// # Arguments
///
/// * `tensor` - The input tensor to convert. Must have shape [channels, height, width].
/// * `out_image_tensor` - The output tensor to store the resulting image. Will be
///   created with shape [height, width, channels] and dtype target_dtype if specified.
/// * `target_dtype` - Optional target dtype for the output image tensor. If not specified, the output dtype will be the same as tensor
P10Error image_from_tensor(const Tensor& tensor, Tensor& out_image_tensor,
                           std::optional<Dtype> target_dtype = std::nullopt
    );
}  // namespace p10::op
