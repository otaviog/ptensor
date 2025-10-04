#include "blur.hpp"

#include <algorithm>
#include <cmath>

#include "tensor.hpp"

namespace p10::op {
namespace {
    void create_gaussian_kernel_(std::span<float> kernel, float sigma);
}  // namespace

PtensorResult<GaussianBlur> GaussianBlur::create(size_t kernel_size, float sigma) {
    if (kernel_size % 2 == 0 || kernel_size > MAX_KERNEL_SIZE) {
        return Err<GaussianBlur>(
            PtensorError::InvalidArgument,
            "Kernel size must be an odd number and less than or equal to "
            "MAX_KERNEL_SIZE."
        );
    }
    KernelStorage kernel_stack_data;
    std::span<float> kernel {kernel_stack_data.data(), kernel_size};
    create_gaussian_kernel_(kernel, sigma);

    return Ok(GaussianBlur(kernel_stack_data, kernel_size));
}

namespace {
    void create_gaussian_kernel_(std::span<float> kernel, float sigma) {
        float sum = 0.0f;
        int half_size = int(kernel.size() / 2);
        for (int i = -half_size; i <= half_size; ++i) {
            kernel[i + half_size] = static_cast<float>(
                std::exp(-(i * i) / (2 * sigma * sigma)) / (sigma * std::sqrt(2 * M_PI))
            );
            sum += kernel[i + half_size];
        }
        // Normalize the kernel
        for (size_t i = 0; i < kernel.size(); ++i) {
            kernel[i] /= sum;
        }
    }

}  // namespace

namespace {
    PtensorResult<PlanarSpan3D<const float>> validate_inputs_(const Tensor& input);
    void apply_horizontal_kernel_(
        Span2D<const float> input,
        Span2D<float> output,
        std::span<const float> kernel
    );
    void apply_vertical_kernel_(
        Span2D<const float> input,
        Span2D<float> output,
        std::span<const float> kernel
    );
}  // namespace

PtensorError GaussianBlur::operator()(const Tensor& input, Tensor& output) const {
    auto validate_result = validate_inputs_(input);
    if (validate_result.is_error()) {
        return validate_result.error();
    }
    const auto input_span {validate_result.unwrap()};

    output.create(input.dtype(), input.shape());
    auto output_span = output.as_planar_span3d<float>().unwrap();

    const auto kernel = get_kernel();
    for (auto channel_plane = 0; channel_plane < input_span.channels(); channel_plane++) {
        apply_horizontal_kernel_(
            input_span.plane(channel_plane),
            output_span.plane(channel_plane),
            kernel
        );
        apply_vertical_kernel_(
            input_span.plane(channel_plane),
            output_span.plane(channel_plane),
            kernel
        );
    }
    return PtensorError::OK;
}

namespace {

    PtensorResult<PlanarSpan3D<const float>> validate_inputs_(const Tensor& input) {
        using ReturnType = PlanarSpan3D<const float>;
        if (input.ndim() != 3) {
            return Err(
                PtensorError::InvalidArgument,
                "Input tensor must be a 3D image with shape (H, W, C)."
            );
        }
        if (input.dtype() != DType::FLOAT32) {
            return Err(PtensorError::InvalidArgument, "Input tensor must be of type UINT8.");
        }

        return input.as_planar_span3d<float>();
    }

    void apply_horizontal_kernel_(
        Span2D<const float> input,
        Span2D<float> output,
        std::span<const float> kernel
    ) {
        int half_size = int(kernel.size()) / 2;
        int height = int(input.height());
        int width = int(input.width());
        for (int y = 0; y < height; ++y) {
            const auto input_row = input.row(y);
            auto output_row = output.row(y);
            for (int x = 0; x < width; ++x) {
                float sum = 0.0f;
                for (int k = -half_size; k <= half_size; ++k) {
                    int xx = std::clamp(x + k, 0, width - 1);
                    sum += input_row[xx] * kernel[k + half_size];
                }
                output_row[x] = sum;
            }
        }
    }

    void apply_vertical_kernel_(
        Span2D<const float> input,
        Span2D<float> output,
        std::span<const float> kernel
    ) {
        int half_size = int(kernel.size()) / 2;
        int height = int(input.height());
        int width = int(input.width());
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float sum = 0.0f;
                for (int k = -half_size; k <= half_size; ++k) {
                    int yy = std::clamp(y + k, 0, height - 1);
                    sum += input.row(yy)[x] * kernel[k + half_size];
                }
                output.row(y)[x] = sum;
            }
        }
    }
}  // namespace
}  // namespace p10::op