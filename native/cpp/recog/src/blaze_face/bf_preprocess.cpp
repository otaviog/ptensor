#include "bf_preprocess.hpp"

#include <algorithm>
#include <array>
#include <tuple>

#include <ptensor/op/resize.hpp>

namespace p10::recog {
constexpr size_t BF_INPUT_CHANNELS = 3;
constexpr std::array<uint8_t, 3> BF_RGB_MEAN = {104, 117, 113};

namespace {
P10Result<std::tuple<size_t, size_t, float>>
    resize(Tensor& input_images, size_t target_size, Tensor& output);

    P10Result<Tensor> create_images_with_padding(
        Tensor& preprocessed,
        int64_t num_images,
        size_t target_size,
        size_t width,
        size_t height
    );

    std::tuple<size_t, size_t, size_t, size_t>
    calculate_padding(size_t target_size, size_t width, size_t height);

    P10Result<Tensor> slice_image_border(Tensor& image, int left, int right, int top, int bottom);

    void subtract_rgb_mean(Tensor& images);
}  // namespace

P10Result<float> BfPreprocessing::process(Tensor& images, Tensor& preprocessed) {
    auto resize_result = resize(images, target_size_, resize_buffer_);
    if (resize_result.is_error()) {
        return Err(resize_result.error());
    }
    const auto [resize_width, resize_height, resize_ratio] = resize_result.unwrap();

    const auto num_images = images.shape().as_span()[0];

    auto inner_preprocessed = create_images_with_padding(
                                  preprocessed,
                                  num_images,
                                  target_size_,
                                  resize_width,
                                  resize_height
    )
                                  .unwrap();
    inner_preprocessed.convert_from(resize_buffer_, TensorOptions(Dtype::Float32));
    subtract_rgb_mean(inner_preprocessed);

    const auto out_shape = preprocessed.shape().as_span();
    preprocessed.reshape(make_shape(
        out_shape[0] / BF_INPUT_CHANNELS,
        BF_INPUT_CHANNELS,
        out_shape[1],
        out_shape[2]
    ));

    return Ok(resize_ratio);
}

namespace {
P10Result<std::tuple<size_t, size_t, float>>
    resize(Tensor& input_images, size_t target_size, Tensor& output) {
        const auto input_shape = input_images.shape().as_span();

        const int64_t num_images = input_shape[0];
        const int64_t input_height = input_shape[2];
        const int64_t input_width = input_shape[3];

        const float original_to_resize_ratio = std::min(
            std::min(
                float(target_size) / float(input_width),
                float(target_size) / float(input_height)
            ),
            1.0f
        );

        size_t resize_width = size_t(std::round(float(input_width) * original_to_resize_ratio));
        size_t resize_height = size_t(std::round(float(input_height) * original_to_resize_ratio));

        Tensor n_by_c_input_images = input_images.as_view();
        n_by_c_input_images.reshape(
            make_shape(num_images * BF_INPUT_CHANNELS, input_height, input_width)
        );
        P10_RETURN_ERR_IF_ERROR(op::resize(n_by_c_input_images, output, resize_width, resize_height)
        );
        return Ok(std::make_tuple(resize_width, resize_height, original_to_resize_ratio));
    }

    P10Result<Tensor> create_images_with_padding(
        Tensor& preprocessed,
        int64_t num_images,
        size_t target_size,
        size_t width,
        size_t height
    ) {
        const auto [left, right, bottom, top] = calculate_padding(target_size, width, height);

        bool new_allocated = false;
        P10_RETURN_ERR_IF_ERROR(preprocessed.create(
            make_shape(
                num_images * BF_INPUT_CHANNELS,
                width + (right - left),
                height + (bottom - top)
            ),
            Dtype::Int8,
            new_allocated
        ));

        if (!new_allocated) {
            preprocessed.fill(0.0);
        }

        return slice_image_border(preprocessed, left, right, top, bottom);
    }

    std::tuple<size_t, size_t, size_t, size_t>
    calculate_padding(size_t target_size, size_t width, size_t height) {
        const float x_pad = float((target_size - width) % 16) * 0.5f;
        const float y_pad = float((target_size - height) % 16) * 0.5f;
        const auto left = size_t(std::round(x_pad - 0.1));
        const auto right = size_t(std::round(x_pad + 0.1));
        const auto top = size_t(std::round(y_pad - 0.1));
        const auto bottom = size_t(std::round(y_pad + 0.1));
        return std::make_tuple(left, right, top, bottom);
    }

    P10Result<Tensor> slice_image_border(Tensor& image, int left, int right, int top, int bottom) {
        const auto shape = image.shape().as_span();
        const auto num_images = shape[0];
        const auto view_width = right - left;
        const auto view_height = bottom - top;

        if (left < 0 || right > shape[1] || top < 0 || bottom > shape[2]) {
            return Err(
                P10Error::InvalidArgument,
                "Border slice indices are out of bounds of the image dimensions"
            );
        }

        auto accessor = image.as_accessor3d<uint8_t>().unwrap();
        auto base_ptr = &accessor[0][top][left];

        return Ok<Tensor>(Tensor::from_data<float>(
            base_ptr,
            make_shape(num_images, int64_t(view_height), int64_t(view_width)),
            image.stride()
        ));
    }

    void subtract_rgb_mean(Tensor& images) {
        auto acc = images.as_accessor3d<float>().unwrap();
        for (size_t channel_idx = 0; channel_idx < acc.channels(); channel_idx += 3) {
            for (size_t k = 0; k < BF_RGB_MEAN.size(); ++k) {
                auto channel = acc[channel_idx + k];
                for (size_t row_idx = 0; row_idx < channel.rows(); ++row_idx) {
                    auto row = channel[row_idx].as_span();
                    std::transform(row.begin(), row.end(), row.begin(), [=](const auto value) {
                        return value - BF_RGB_MEAN[k];
                    });
                }
            }
        }
    }
}  // namespace

}  // namespace p10::recog
