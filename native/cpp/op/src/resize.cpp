#include "ptensor/op/resize.hpp"

#include "ptensor/dtype.hpp"
#include "ptensor/tensor.hpp"

namespace p10::op {
PtensorError resize(const Tensor& input, Tensor& output, size_t new_width, size_t new_height) {
    auto input_validate = input.as_planar_span3d<float>();
    if (input_validate.is_error()) {
        return input_validate.unwrap_err();
    }

    const auto input_span = input_validate.unwrap();
    const auto channels = input_span.channels();
    const auto height = input_span.height();
    const auto width = input_span.width();

    const float x_scale = float(width) / float(new_width);
    const float y_scale = float(height) / float(new_height);

    if (auto err = output.create(
            make_shape(int64_t(channels), int64_t(new_height), int64_t(new_width)),
            Dtype::Float32
        );
        err.is_error()) {
        return err;
    }

    auto output_span = output.as_planar_span3d<float>().unwrap();

    for (int row = 0; row < new_height; ++row) {
        const auto src_y = std::min(size_t(float(row) * y_scale), height - 1);

        for (int col = 0; col < new_width; ++col) {
            const auto src_x = std::min(size_t(float(col) * x_scale), width - 1);

            for (size_t chn = 0; chn < channels; ++chn) {
                output_span.plane(chn).row(row)[col] = input_span.plane(chn).row(src_y)[src_x];
            }
        }
    }
    return PtensorError::Ok;
}

}  // namespace p10::op