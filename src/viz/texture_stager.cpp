#include "texture_stager.hpp"

#include <algorithm>
#include <format>

#include <ptensor/tensor.hpp>

namespace p10::viz {

namespace {

/// Format a tensor maps to from its channel count, before clamping to what
/// the backend supports.
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
    return static_cast<uint8_t>(value * 255.0F);
}

/// Convert source pixels (any layout/dtype/channel count) into `target`,
/// writing into `buf`. The buffer only grows, so it is reused across uploads.
template<typename Src>
void convert(const Src* src, int64_t height, int64_t width, int64_t channels, bool hwc,
             TextureFormat target, std::vector<uint8_t>& buf) {
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

bool TextureStager::supports(TextureFormat format) const {
    return std::ranges::find(supported_, format) != supported_.end();
}

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

    const TextureFormat natural = natural_format(channels);
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
        convert(src.unwrap().data(), height, width, channels, hwc, target, buffer_);
    } else {
        return Err(
            P10Error::InvalidArgument << "Unsupported dtype: " << to_string(tensor.dtype())
        );
    }

    view.data = buffer_.data();
    view.size_bytes = static_cast<size_t>(height * width) * stride;
    return Ok(view);
}

}  // namespace p10::viz
