#pragma once

#include <string>

#include <ptensor/tensor.hpp>

namespace p10::io {

/// Loads an image from disk.
///
/// Decodes the image using stb_image and returns a tensor with shape
/// (height, width, channels) in Uint8 dtype.
///
/// # Arguments
///
/// * `path`: File path to load. Supports common formats (PNG, JPG, BMP, etc.)
///   as handled by stb_image.
///
/// # Returns
///
/// * Tensor with shape (height, width, channels) in Uint8 dtype on success.
///
/// # Errors
///
/// * InvalidArgument: File not found or decoding failed.
P10Result<Tensor> load_image(const std::string& path);

/// Saves a tensor as a PNG image to disk.
///
/// Accepts 2D tensors (grayscale) or 3D tensors (RGB/RGBA).
/// Both must be Uint8 dtype.
///
/// # Arguments
///
/// * `path`: Output file path (extension is ignored; PNG format is always used).
/// * `tensor`: 2D (height, width) or 3D (height, width, channels) Uint8 tensor.
///
/// # Errors
///
/// * InvalidArgument: Tensor is not Uint8 dtype or has unsupported dimensions.
P10Error save_image(const std::string& path, const Tensor& tensor);

}  // namespace p10::io
