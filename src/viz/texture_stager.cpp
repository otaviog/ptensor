#include "texture_stager.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>

#include <ptensor/tensor.hpp>

namespace p10::viz {

namespace {

    TextureFormat natural_format(int64_t channels);
    size_t pixel_stride(TextureFormat format);
    uint8_t to_u8(uint8_t value);
    uint8_t to_u8(float value);
    void jet(float t, uint8_t& r, uint8_t& g, uint8_t& b);
    void convert_heatmap(const float* src, int64_t npixels, std::vector<uint8_t>& buf);

    template<typename Src>
    void convert(
        const Src* src,
        int64_t height,
        int64_t width,
        int64_t channels,
        bool hwc,
        TextureFormat target,
        std::vector<uint8_t>& buf
    );

}  // namespace

bool TextureStager::supports(TextureFormat format) const {
    return std::ranges::find(supported_, format) != supported_.end();
}

/// # Arguments
///
/// * `tensor`: rank-3 image tensor ([H,W,C] or [C,H,W]), dtype UInt8 or Float32,
///   1, 3, or 4 channels.
/// * `layout`: whether `tensor` is in HWC or CHW memory order.
///
/// # Returns
///
/// * `UploadView` pointing directly into `tensor` (zero copy) when the tensor's
///   natural format is already supported, or into an internal conversion buffer
///   otherwise.  Float single-channel tensors are tone-mapped to an RGBA jet
///   heatmap, normalized over the finite min/max of the frame.  Float RGB(A)
///   tensors are scaled from [0, 1] to [0, 255] with clamp and round.
///
/// # Errors
///
/// * `P10Error::InvalidArgument` — tensor is not rank-3, channel count is not
///   1/3/4, or dtype is not UInt8/Float32.
P10Result<UploadView> TextureStager::stage(const Tensor& tensor, TensorLayout layout) {
    if (tensor.dims() != 3) {
        return Err(
            P10Error::InvalidArgument
            << std::format("Tensor must have rank 3, got rank {}", tensor.dims())
        );
    }

    const bool hwc = (layout == TensorLayout::HWC);
    const int64_t height = hwc ? tensor.shape()[0].unwrap() : tensor.shape()[1].unwrap();
    const int64_t width = hwc ? tensor.shape()[1].unwrap() : tensor.shape()[2].unwrap();
    const int64_t channels = hwc ? tensor.shape()[2].unwrap() : tensor.shape()[0].unwrap();

    if (channels != 1 && channels != 3 && channels != 4) {
        return Err(
            P10Error::InvalidArgument
            << std::format("Channels must be 1, 3, or 4, got {}", channels)
        );
    }

    // Single-channel float images are shown as a heatmap (always RGBA) so small
    // differences are visible; u8 gray and multi-channel keep their natural format.
    const bool heatmap = (tensor.dtype() == Dtype::Float32 && channels == 1);
    const TextureFormat natural = heatmap ? TextureFormat::Rgba8 : natural_format(channels);
    const TextureFormat target = supports(natural) ? natural : TextureFormat::Rgba8;

    UploadView view;
    view.format = target;
    view.width = static_cast<int>(width);
    view.height = static_cast<int>(height);

    const size_t stride = pixel_stride(target);

    // Fast path: UInt8 HWC whose interleaved bytes already match the target.
    if (tensor.dtype() == Dtype::Uint8 && hwc && channels == static_cast<int64_t>(stride)) {
        auto src = tensor.as_span1d<uint8_t>();
        if (src.is_error()) {
            return Err(src);
        }
        view.data = src.unwrap().data();
        view.size_bytes = static_cast<size_t>(height * width) * stride;
        return Ok(view);
    }

    if (tensor.dtype() == Dtype::Uint8) {
        auto src = tensor.as_span1d<uint8_t>();
        if (src.is_error()) {
            return Err(src);
        }
        convert(src.unwrap().data(), height, width, channels, hwc, target, buffer_);
    } else if (tensor.dtype() == Dtype::Float32) {
        auto src = tensor.as_span1d<float>();
        if (src.is_error()) {
            return Err(src);
        }
        if (heatmap) {
            convert_heatmap(src.unwrap().data(), height * width, buffer_);
        } else {
            convert(src.unwrap().data(), height, width, channels, hwc, target, buffer_);
        }
    } else {
        return Err(P10Error::InvalidArgument << "Unsupported dtype: " << to_string(tensor.dtype()));
    }

    view.data = buffer_.data();
    view.size_bytes = static_cast<size_t>(height * width) * stride;
    return Ok(view);
}

namespace {

    TextureFormat natural_format(int64_t channels) {
        return channels == 1 ? TextureFormat::Gray8 : TextureFormat::Rgba8;
    }

    size_t pixel_stride(TextureFormat format) {
        return format == TextureFormat::Gray8 ? 1 : 4;
    }

    uint8_t to_u8(uint8_t value) {
        return value;
    }

    uint8_t to_u8(float value) {
        // RGB(x) float channels are treated as normalized [0, 1]; clamp so
        // out-of-range values saturate instead of wrapping mod 256, then round.
        const float scaled = std::clamp(value * 255.0F, 0.0F, 255.0F);
        return static_cast<uint8_t>(std::lround(scaled));
    }

    void jet(float t, uint8_t& r, uint8_t& g, uint8_t& b) {
        auto ch = [](float x) {
            return static_cast<uint8_t>(std::lround(std::clamp(x, 0.0F, 1.0F) * 255.0F));
        };
        r = ch(1.5F - std::fabs(4.0F * t - 3.0F));
        g = ch(1.5F - std::fabs(4.0F * t - 2.0F));
        b = ch(1.5F - std::fabs(4.0F * t - 1.0F));
    }

    void convert_heatmap(const float* src, int64_t npixels, std::vector<uint8_t>& buf) {
        const size_t stride = 4;  // Heatmap output is always RGBA.
        if (buf.size() < static_cast<size_t>(npixels) * stride) {
            buf.resize(static_cast<size_t>(npixels) * stride);
        }

        float lo = std::numeric_limits<float>::infinity();
        float hi = -std::numeric_limits<float>::infinity();
        for (int64_t px = 0; px < npixels; ++px) {
            const float v = src[px];
            if (std::isfinite(v)) {
                lo = std::min(lo, v);
                hi = std::max(hi, v);
            }
        }
        const float range = hi - lo;

        for (int64_t px = 0; px < npixels; ++px) {
            const float v = src[px];
            const float t = (range > 0.0F && std::isfinite(v)) ? (v - lo) / range : 0.0F;
            uint8_t* dst = buf.data() + static_cast<size_t>(px) * stride;
            jet(t, dst[0], dst[1], dst[2]);
            dst[3] = 255;
        }
    }

    template<typename Src>
    void convert(
        const Src* src,
        int64_t height,
        int64_t width,
        int64_t channels,
        bool hwc,
        TextureFormat target,
        std::vector<uint8_t>& buf
    ) {
        const int64_t npixels = height * width;
        const size_t stride = pixel_stride(target);
        if (buf.size() < static_cast<size_t>(npixels) * stride) {
            buf.resize(static_cast<size_t>(npixels) * stride);
        }

        auto sample = [&](int64_t px, int64_t c) -> uint8_t {
            const int64_t idx = hwc ? px * channels + c : c * npixels + px;
            return to_u8(src[idx]);
        };

        for (int64_t px = 0; px < npixels; ++px) {
            uint8_t* dst = buf.data() + static_cast<size_t>(px) * stride;
            if (target == TextureFormat::Gray8) {
                dst[0] = sample(px, 0);
            } else {
                // Single-channel sources broadcast across RGB; missing alpha is opaque.
                dst[0] = sample(px, 0);
                dst[1] = sample(px, channels == 1 ? 0 : 1);
                dst[2] = sample(px, channels == 1 ? 0 : 2);
                dst[3] = channels == 4 ? sample(px, 3) : 255;
            }
        }
    }

}  // namespace

}  // namespace p10::viz
