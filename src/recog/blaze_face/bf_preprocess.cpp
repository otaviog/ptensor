#include "bf_preprocess.hpp"

#include <algorithm>
#include <array>
#include <tuple>

#include <ptensor/op/resize.hpp>

namespace p10::recog {
constexpr size_t BF_INPUT_CHANNELS = 3;
// Channel means in BGR order: subtracted from the model's channel 0 (B),
// channel 1 (G), channel 2 (R) respectively. The original RFB / BlazeFace
// training pipeline uses OpenCV's BGR layout, so input RGB pixels must be
// swapped before subtracting.
constexpr std::array<float, 3> BF_BGR_MEAN = {104.0f, 117.0f, 123.0f};

namespace {
    P10Result<std::tuple<size_t, size_t, float>>
    resize(Tensor& input_images, size_t target_size, Tensor& output);

    std::tuple<size_t, size_t, size_t, size_t>
    calculate_padding(size_t target_size, size_t width, size_t height);

    void
    convert_rgb_to_bgr_mean_into_padded(const Tensor& src, Tensor& dst, size_t top, size_t left);
}  // namespace

P10Result<float> BfPreprocessing::process(Tensor& images, Tensor& preprocessed) {
    auto resize_result = resize(images, target_size_, resize_buffer_);
    if (resize_result.is_error()) {
        return Err(resize_result.error());
    }
    const auto [resize_width, resize_height, resize_ratio] = resize_result.unwrap();

    const auto num_images = images.shape().as_span()[0];
    const auto [left, right, top, bottom] =
        calculate_padding(target_size_, resize_width, resize_height);
    const auto padded_h = resize_height + top + bottom;
    const auto padded_w = resize_width + left + right;

    bool new_allocated = false;
    preprocessed.create(
        make_shape(num_images * int64_t(BF_INPUT_CHANNELS), int64_t(padded_h), int64_t(padded_w)),
        Dtype::Float32,
        new_allocated
    );
    if (new_allocated) {
        preprocessed.fill(0.0f);
    }

    convert_rgb_to_bgr_mean_into_padded(resize_buffer_, preprocessed, top, left);

    const auto out_shape = preprocessed.shape().as_span();
    preprocessed.reshape(
        make_shape(out_shape[0] / BF_INPUT_CHANNELS, BF_INPUT_CHANNELS, out_shape[1], out_shape[2])
    );

    return Ok(static_cast<float>(resize_ratio));
}

namespace {
    P10Result<std::tuple<size_t, size_t, float>>
    resize(Tensor& input_images, size_t target_size, Tensor& output) {
        const auto input_shape = input_images.shape().as_span();

        const int64_t num_images = input_shape[0];
        const int64_t input_height = input_shape[2];
        const int64_t input_width = input_shape[3];

        const float original_to_resize_ratio = std::min(
            {static_cast<float>(target_size) / static_cast<float>(input_width),
             static_cast<float>(target_size) / static_cast<float>(input_height),
             1.0f}
        );

        const auto resize_width =
            static_cast<size_t>(std::round(float(input_width) * original_to_resize_ratio));
        const auto resize_height =
            static_cast<size_t>(std::round(float(input_height) * original_to_resize_ratio));

        Tensor n_by_c_input_images = input_images.as_view();
        n_by_c_input_images.reshape(
            make_shape(num_images * BF_INPUT_CHANNELS, input_height, input_width)
        );
        P10_RETURN_ERR_IF_ERROR(
            op::resize(n_by_c_input_images, output, resize_width, resize_height)
        );
        return Ok(std::make_tuple(resize_width, resize_height, original_to_resize_ratio));
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

    /// Converts the resized RGB uint8 planes into the padded float BGR-mean
    /// output, writing directly at the correct strided positions.
    void
    convert_rgb_to_bgr_mean_into_padded(const Tensor& src, Tensor& dst, size_t top, size_t left) {
        const auto src_acc = src.as_accessor3d<const uint8_t>().unwrap();
        auto dst_acc = dst.as_accessor3d<float>().unwrap();

        // src shape: [N*3, resize_h, resize_w] (planar RGB per image)
        // dst shape: [N*3, padded_h, padded_w]
        // For each group of 3 planes (R, G, B), write BGR-mean into dst
        // at the offset (top, left).
        for (size_t ch_idx = 0; ch_idx + 2 < src_acc.channels(); ch_idx += 3) {
            const auto src_r = src_acc[ch_idx + 0];
            const auto src_g = src_acc[ch_idx + 1];
            const auto src_b = src_acc[ch_idx + 2];

            auto dst_0 = dst_acc[ch_idx + 0];  // output B-mean plane
            auto dst_1 = dst_acc[ch_idx + 1];  // output G-mean plane
            auto dst_2 = dst_acc[ch_idx + 2];  // output R-mean plane

            for (size_t row = 0; row < src_r.rows(); ++row) {
                auto dst_row_0 = dst_0[top + row].as_span();
                auto dst_row_1 = dst_1[top + row].as_span();
                auto dst_row_2 = dst_2[top + row].as_span();

                for (size_t col = 0; col < src_r.cols(); ++col) {
                    dst_row_0[left + col] = float(src_b[row][col]) - BF_BGR_MEAN[0];
                    dst_row_1[left + col] = float(src_g[row][col]) - BF_BGR_MEAN[1];
                    dst_row_2[left + col] = float(src_r[row][col]) - BF_BGR_MEAN[2];
                }
            }
        }
    }
}  // namespace

}  // namespace p10::recog
