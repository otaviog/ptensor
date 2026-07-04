#include <algorithm>

#include <ptensor/op/image_layout.hpp>
#include <ptensor/tensor.hpp>

namespace p10::op {

P10Error
image_to_tensor(const Tensor& image, Tensor& out_tensor, const ImageToTensorOptions& options) {
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

    // HWC image: dims come from the tensor shape (Accessor3D names dims for the
    // CHW case, so don't query channels()/rows()/cols() here).
    const auto image_span = image.as_accessor3d<uint8_t>().expect("Invalid image");
    const auto height = static_cast<size_t>(image.shape(0).unwrap());
    const auto width = static_cast<size_t>(image.shape(1).unwrap());
    const auto num_channels = static_cast<size_t>(image.shape(2).unwrap());
    const Dtype out_dtype = options.target_dtype().value_or(Dtype::Float32);
    const bool do_normalize = options.normalize();

    out_tensor.create(
        make_shape(
            static_cast<int64_t>(num_channels),
            static_cast<int64_t>(height),
            static_cast<int64_t>(width)
        ),
        out_dtype
    );

    out_dtype.match(
        [&](auto int_id) {
            using T = decltype(int_id)::type;
            auto out_span = out_tensor.as_span3d<T>().unwrap();
            for (size_t row = 0; row < height; row++) {
                for (size_t col = 0; col < width; col++) {
                    const auto& ch = image_span[row][col];
                    for (size_t c = 0; c < num_channels; c++) {
                        out_span[c][row][col] = static_cast<T>(ch[c]);
                    }
                }
            }
        },
        [&](auto float_id) {
            using T = decltype(float_id)::type;
            auto out_span = out_tensor.as_span3d<T>().unwrap();
            for (size_t row = 0; row < height; row++) {
                for (size_t col = 0; col < width; col++) {
                    const auto& ch = image_span[row][col];
                    for (size_t c = 0; c < num_channels; c++) {
                        out_span[c][row][col] = do_normalize ? T(ch[c]) / T(255) : T(ch[c]);
                    }
                }
            }
        }
    );

    if (options.unsqueeze()) {
        out_tensor.reshape(make_shape(
            int64_t {1},
            static_cast<int64_t>(num_channels),
            static_cast<int64_t>(height),
            static_cast<int64_t>(width)
        ));
    }

    return P10Error::Ok;
}

P10Error image_from_tensor(
    const Tensor& tensor,
    Tensor& out_image_tensor,
    const ImageFromTensorOptions& options
) {
    if (tensor.dtype() != Dtype::Float32 && tensor.dtype() != Dtype::Float64) {
        return P10Error::InvalidArgument << "Input tensor must be of float type.";
    }
    const size_t dims = tensor.shape().dims();
    if (dims != 3 && dims != 4) {
        return P10Error::InvalidArgument
            << "Input tensor must have shape [C, H, W] or [1, C, H, W].";
    }
    if (dims == 4 && tensor.shape(0).unwrap() != 1) {
        return P10Error::InvalidArgument
            << "4D input must have a leading batch dimension of size 1.";
    }
    if (!tensor.is_contiguous()) {
        return P10Error::InvalidArgument << "Input tensor must be contiguous in memory.";
    }

    const size_t base = (dims == 4) ? 1 : 0;
    const auto num_channels = static_cast<size_t>(tensor.shape(base).unwrap());
    const auto height = static_cast<size_t>(tensor.shape(base + 1).unwrap());
    const auto width = static_cast<size_t>(tensor.shape(base + 2).unwrap());
    const Dtype out_dtype = options.target_dtype().value_or(Dtype::Uint8);
    const bool do_normalize = options.normalize();

    out_image_tensor.create(
        make_shape(
            static_cast<int64_t>(height),
            static_cast<int64_t>(width),
            static_cast<int64_t>(num_channels)
        ),
        out_dtype
    );

    tensor.dtype().match(
        // The dtype check above guarantees this branch is never taken; it
        // exists only to satisfy Dtype::match's signature.
        [](auto /*int_id*/) {},
        [&](auto float_id) {
            using Fin = decltype(float_id)::type;
            // Input is contiguous and either [C, H, W] or [1, C, H, W] — both
            // share the same planar memory layout.
            const auto* in_data = reinterpret_cast<const Fin*>(tensor.as_bytes().data());
            const size_t plane_size = height * width;
            out_dtype.match(
                [&](auto int_out_id) {
                    using Tout = decltype(int_out_id)::type;
                    auto out_span = out_image_tensor.as_accessor3d<Tout>().unwrap();
                    for (size_t row = 0; row < height; row++) {
                        for (size_t col = 0; col < width; col++) {
                            auto out_ch = out_span[row][col];
                            for (size_t c = 0; c < num_channels; c++) {
                                const Fin in_val = in_data[(c * plane_size) + (row * width) + col];
                                const Fin val = do_normalize ? in_val * Fin {255} : in_val;
                                out_ch[c] = static_cast<Tout>(std::clamp(val, Fin {0}, Fin {255}));
                            }
                        }
                    }
                },
                [&](auto float_out_id) {
                    using Tout = decltype(float_out_id)::type;
                    auto out_span = out_image_tensor.as_accessor3d<Tout>().unwrap();
                    for (size_t row = 0; row < height; row++) {
                        for (size_t col = 0; col < width; col++) {
                            auto out_ch = out_span[row][col];
                            for (size_t c = 0; c < num_channels; c++) {
                                out_ch[c] = static_cast<Tout>(
                                    in_data[(c * plane_size) + (row * width) + col]
                                );
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
