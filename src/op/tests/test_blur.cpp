#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <ptensor/io/image.hpp>
#include <ptensor/op/blur.hpp>
#include <ptensor/op/image_layout.hpp>
#include <ptensor/tensor.hpp>
#include <ptensor/testing/catch2_assertions.hpp>
#include <ptensor/testing/compare_tensors.hpp>

#include "testing.hpp"

namespace p10::op {
TEST_CASE("Op: Blur image", "[tensorop][blur][integration]") {
    auto [sample_image, image_file] = testing::samples::image01();
    Tensor sample_tensor;
    REQUIRE(op::image_to_tensor(sample_image, sample_tensor) == P10Error::Ok);
    const auto test_output_path = testing::get_output_path("op/blur");

    SECTION("Apply blur operator") {
        auto blur_op = GaussianBlur::create(25, 1.5f).unwrap();
        Tensor blurred_tensor;
        REQUIRE(blur_op.transform(sample_tensor, blurred_tensor).is_ok());
        REQUIRE(blurred_tensor.shape() == sample_tensor.shape());

        Tensor blurred_image;
        REQUIRE(op::image_from_tensor(blurred_tensor, blurred_image) == P10Error::Ok);
        REQUIRE(
            io::save_image(
                (test_output_path / testing::suffixed(image_file, "blur")).string(),
                blurred_image
            )
                .is_ok()
        );
    }

    SECTION("Blurring a constant image keeps the constant") {
        // A normalized Gaussian kernel sums to 1 and edges are clamped, so a
        // uniform image is unchanged by the blur.
        const auto constant = Tensor::full(make_shape(3, 40, 40), 0.5).unwrap();
        auto blur_op = GaussianBlur::create(9, 1.5f).unwrap();
        Tensor blurred;
        REQUIRE(blur_op.transform(constant, blurred).is_ok());
        REQUIRE_THAT(
            testing::compare_tensors(constant, blurred, testing::CompareOptions().tolerance(1e-4)),
            testing::is_ok()
        );
    }

    SECTION("Should fail with invalid kernel size") {
        REQUIRE(GaussianBlur::create(GaussianBlur::MAX_KERNEL_SIZE + 1, 1.5f).is_error());
        REQUIRE(GaussianBlur::create(0, 1.5f).is_error());
    }
}

namespace {
    std::vector<float> gaussian_1d(int kernel_size, float sigma) {
        const int half = kernel_size / 2;
        std::vector<float> kernel(kernel_size);
        float sum = 0.0F;
        for (int i = -half; i <= half; ++i) {
            kernel[i + half] = std::exp(-(i * i) / (2 * sigma * sigma))
                / (sigma * std::sqrt(2.0F * std::numbers::pi));
            sum += kernel[i + half];
        }
        for (auto& value : kernel) {
            value /= sum;
        }
        return kernel;
    }
}  // namespace

// The fast path (2D float32, kernel 3/5/7/9) runs a simd-tiled horizontal pass,
// a transpose, another horizontal pass, and a final transpose. Check it against
// a naive clamped separable convolution (horizontal then vertical).
TEST_CASE("Op: Blur float32 plane", "[tensorop][blur]") {
    const int height = 67;  // non-multiples of the SIMD block to exercise borders
    const int width = 83;
    const float sigma = 1.5F;
    const auto kernel_size = GENERATE(3, 5, 7, 9);

    DYNAMIC_SECTION("kernel size " << kernel_size) {
        std::mt19937_64 const rng(123);
        const Tensor input = Tensor::from_random(
                                 make_shape(height, width),
                                 rng,
                                 TensorOptions().dtype(Dtype::Float32)
        )
                                 .unwrap();

        auto blur_op = GaussianBlur::create(kernel_size, sigma).unwrap();
        Tensor output;
        REQUIRE(blur_op.transform(input, output).is_ok());
        REQUIRE(output.shape() == input.shape());

        const auto kernel = gaussian_1d(kernel_size, sigma);
        const int half = kernel_size / 2;
        const auto in = input.as_span2d<const float>().unwrap();

        std::vector<float> horizontal(static_cast<size_t>(height * width));
        std::vector<float> reference(static_cast<size_t>(height * width));
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float sum = 0.0F;
                for (int k = -half; k <= half; ++k) {
                    sum += in[y][std::clamp(x + k, 0, width - 1)] * kernel[k + half];
                }
                horizontal[(y * width) + x] = sum;
            }
        }
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float sum = 0.0F;
                for (int k = -half; k <= half; ++k) {
                    sum += horizontal[(std::clamp(y + k, 0, height - 1) * width) + x]
                        * kernel[k + half];
                }
                reference[(y * width) + x] = sum;
            }
        }

        const auto out = output.as_span2d<const float>().unwrap();
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                CAPTURE(y, x);
                REQUIRE(out[y][x] == Catch::Approx(reference[(y * width) + x]).margin(1e-4));
            }
        }
    }
}
}  // namespace p10::op
