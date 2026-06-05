#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include <ptensor/p10_result.hpp>

#include "image_texture.hpp"  // TensorLayout

namespace p10 {
class Tensor;
}

namespace p10::viz {

/// Pixel format of a GPU texture a backend can sample.
enum class TextureFormat {
    Gray8,  ///< one 8-bit channel (R8)
    Rgba8,  ///< four 8-bit channels (RGBA)
};

/// Upload-ready pixels plus the texture shape they describe.
struct UploadView {
    TextureFormat format = TextureFormat::Rgba8;
    int width = 0;
    int height = 0;
    const void* data = nullptr;
    size_t size_bytes = 0;
};

/// Stages a tensor into a pixel format the graphics backend supports.
///
/// Construct with the formats the backend can sample natively. `stage`
/// returns a view of upload-ready pixels: a direct pointer into the tensor
/// when its natural format is already supported (zero copy), otherwise a
/// pointer into a reused internal buffer holding the converted pixels.
class TextureStager {
  public:
    explicit TextureStager(std::initializer_list<TextureFormat> supported)
        : supported_(supported) {}

    /// Stage `tensor` ([H,W,C] or [C,H,W], C in {1,3,4}, UInt8 or Float32).
    P10Result<UploadView> stage(const Tensor& tensor, TensorLayout layout);

  private:
    bool supports(TextureFormat format) const;

    std::vector<TextureFormat> supported_;
    std::vector<uint8_t> buffer_;
};

}  // namespace p10::viz
