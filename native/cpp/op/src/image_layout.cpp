#include <ptensor/op/image_layout.hpp>

#include <algorithm>

#include <ptensor/tensor.hpp>

namespace p10::op {

P10Error image_to_tensor(
    const Tensor& image,
    Tensor& out_tensor,
    std::optional<Dtype> target_dtype,
    ImageToTensorNormalize normalize_opt,
    ImageToTensorSqueeze squeeze_opt
) {
    if (image.dtype() != Dtype::Uint8) {
        return P10Error::InvalidArgument << "Input tensor must be of type UINT8.";
    }
    if (image.shape().dims() != 3) {
        return P10Error::InvalidArgument
            << "Input tensor must have shape [height, width, channels].";
    }
    if (!image.is_contiguous()) {
        return P10Error::InvalidArgument << "Input tensor must be contiguous in memory.";
    }

    const auto image_span = image.as_span3d<uint8_t>().expect("Invalid image");
    const size_t num_channels = image_span.channels();
    const size_t height = image_span.height();
    const size_t width = image_span.width();
    const Dtype out_dtype = target_dtype.value_or(Dtype::Float32);

    out_tensor.create(
        make_shape(int64_t(num_channels), int64_t(height), int64_t(width)), out_dtype
    );

    out_dtype.match(
        [&](auto int_id) {
            using T = typename decltype(int_id)::type;
            auto out_span = out_tensor.as_planar_span3d<T>().unwrap();
            for (size_t row = 0; row < height; row++) {
                for (size_t col = 0; col < width; col++) {
                    const auto& ch = image_span.channel(row, col);
                    for (size_t c = 0; c < num_channels; c++) {
                        out_span[c].row(row)[col] = static_cast<T>(ch[c]);
                    }
                }
            }
        },
        [&](auto float_id) {
            using T = typename decltype(float_id)::type;
            auto out_span = out_tensor.as_planar_span3d<T>().unwrap();
            const bool do_normalize = normalize_opt == ImageToTensorNormalize::Normalize;
            for (size_t row = 0; row < height; row++) {
                for (size_t col = 0; col < width; col++) {
                    const auto& ch = image_span.channel(row, col);
                    for (size_t c = 0; c < num_channels; c++) {
                        out_span[c].row(row)[col] = do_normalize ? T(ch[c]) / T(255) : T(ch[c]);
                    }
                }
            }
        }
    );

    if (squeeze_opt == ImageToTensorSqueeze::Squeeze) {
        out_tensor.squeeze();
    }

    return P10Error::Ok;
}

P10Error image_from_tensor(
    const Tensor& tensor,
    Tensor& out_image_tensor,
    std::optional<Dtype> target_dtype
) {
    if (tensor.shape().dims() != 3) {
        return P10Error::InvalidArgument
            << "Input tensor must have shape [channels, height, width].";
    }
    if (!tensor.is_contiguous()) {
        return P10Error::InvalidArgument << "Input tensor must be contiguous in memory.";
    }

    const size_t num_channels = size_t(tensor.shape(0).unwrap());
    const size_t height = size_t(tensor.shape(1).unwrap());
    const size_t width = size_t(tensor.shape(2).unwrap());
    const Dtype out_dtype = target_dtype.value_or(Dtype::Uint8);

    out_image_tensor.create(
        make_shape(int64_t(height), int64_t(width), int64_t(num_channels)), out_dtype
    );

    tensor.dtype().match(
        [&](auto int_id) {
            // integer input: cast directly to output type
            using Tin = typename decltype(int_id)::type;
            auto in_span = tensor.as_planar_span3d<Tin>().unwrap();
            out_dtype.match([&](auto out_id) {
                using Tout = typename decltype(out_id)::type;
                auto out_span = out_image_tensor.as_span3d<Tout>().unwrap();
                for (size_t row = 0; row < height; row++) {
                    for (size_t col = 0; col < width; col++) {
                        auto out_ch = out_span.channel(row, col);
                        for (size_t c = 0; c < num_channels; c++) {
                            out_ch[c] = static_cast<Tout>(in_span[c].row(row)[col]);
                        }
                    }
                }
            });
        },
        [&](auto float_id) {
            using Fin = typename decltype(float_id)::type;
            auto in_span = tensor.as_planar_span3d<Fin>().unwrap();
            out_dtype.match(
                [&](auto int_out_id) {
                    // float → integer: scale [0,1] to [0,255]
                    using Tout = typename decltype(int_out_id)::type;
                    auto out_span = out_image_tensor.as_span3d<Tout>().unwrap();
                    for (size_t row = 0; row < height; row++) {
                        for (size_t col = 0; col < width; col++) {
                            auto out_ch = out_span.channel(row, col);
                            for (size_t c = 0; c < num_channels; c++) {
                                Fin val = in_span[c].row(row)[col] * Fin(255);
                                out_ch[c] = static_cast<Tout>(std::clamp(val, Fin(0), Fin(255)));
                            }
                        }
                    }
                },
                [&](auto float_out_id) {
                    using Tout = typename decltype(float_out_id)::type;
                    auto out_span = out_image_tensor.as_span3d<Tout>().unwrap();
                    for (size_t row = 0; row < height; row++) {
                        for (size_t col = 0; col < width; col++) {
                            auto out_ch = out_span.channel(row, col);
                            for (size_t c = 0; c < num_channels; c++) {
                                out_ch[c] = static_cast<Tout>(in_span[c].row(row)[col]);
                            }
                        }
                    }
                }
            );
        }
    );

    return P10Error::Ok;
}

}  // namespace p10::op
