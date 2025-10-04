#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <p10/io/image.hpp>
#include <p10/tensor.hpp>

#include "imageop/image_to_tensor.hpp"
#include "tensorop/blur.hpp"
#include "tensorop/elemwise.hpp"
#include "tensorop/resize.hpp"
#include "testing.hpp"

namespace p10::tensorop {
using Catch::Approx;

TEST_CASE("Tensorop: Add", "[tensorop]") {
    auto type = GENERATE(DType::FLOAT32, DType::INT64, DType::UINT8);
    DYNAMIC_SECTION("Testing addition with type " << DType(type).to_string()) {
        auto a = Tensor::from_range(type, {2, 3});
        auto b = Tensor::from_range(type, {2, 3});
        Tensor out;
        REQUIRE(add_elemwise(a, b, out).is_ok());
        REQUIRE(out.shape(0) == 2);
        REQUIRE(out.shape(1) == 3);
        REQUIRE(out.size() == 6);
        REQUIRE(out.dtype() == type);

        out.visit_data([](auto span) {
            using SpanType = decltype(span)::value_type;
            for (int i = 0; i < 6; ++i) {
                REQUIRE(span[i] == Approx(static_cast<SpanType>(i * 2)));
            }
        });
    }
}

TEST_CASE("Tensorop: Subtract", "[tensor]") {
    auto type = GENERATE(DType::FLOAT32, DType::INT64, DType::UINT8);
    DYNAMIC_SECTION("Testing subtraction with type " << DType(type).to_string()) {
        auto a = Tensor::from_range(type, {2, 3});
        auto b = Tensor::from_range(type, {2, 3});
        Tensor out;
        REQUIRE(subtract_elemwise(a, b, out).is_ok());
        REQUIRE(out.shape(0) == 2);
        REQUIRE(out.shape(1) == 3);
        REQUIRE(out.size() == 6);
        REQUIRE(out.dtype() == type);

        out.visit_data([](auto span) {
            using SpanType = decltype(span)::value_type;
            for (int i = 0; i < 6; ++i) {
                REQUIRE(span[i] == Approx(0.0));
            }
        });
    }
}

TEST_CASE("Tensorop: Blur image", "[tensorop]") {
    auto sample_image = testing::samples::image01();
    Tensor sample_tensor;

    imageop::image_to_tensor(sample_image, sample_tensor);
    SECTION("Apply blur operator") {
        auto blur_op = GaussianBlur::create(25, 1.5f).unwrap();
        Tensor blurred_tensor;
        blur_op(sample_tensor, blurred_tensor).panic("Failed to blur image");

        REQUIRE(blurred_tensor.shape() == sample_tensor.shape());

        Tensor blurred_image;
        imageop::image_from_tensor(blurred_tensor, blurred_image);
        io::save_image((testing::get_output_path() / "000001_blurred.jpg").string(), blurred_image);
    }

    SECTION("Should fail with invalid kernel size") {
        REQUIRE(GaussianBlur::create(GaussianBlur::MAX_KERNEL_SIZE + 1, 1.5f).is_error());
        REQUIRE(GaussianBlur::create(0, 1.5f).is_error());
    }
}

TEST_CASE("Tensorop: Resize", "[tensorop]") {
    auto sample_image = testing::samples::image01();
    Tensor sample_tensor;
    imageop::image_to_tensor(sample_image, sample_tensor);
    SECTION("Downsample image") {
        Tensor resized_tensor;
        REQUIRE(resize(sample_tensor, resized_tensor, 128, 128).is_ok());
        REQUIRE(resized_tensor.shape() == std::vector<int64_t>({3, 128, 128}));
        REQUIRE(resized_tensor.dtype() == DType::FLOAT32);
        Tensor resized_image;
        imageop::image_from_tensor(resized_tensor, resized_image);
        io::save_image(
            (testing::get_output_path() / "000001_resized_downsampled.jpg").string(),
            resized_image
        );
    }
    SECTION("Upsample image") {
        Tensor resized_tensor;
        REQUIRE(resize(sample_tensor, resized_tensor, 1024, 1024).is_ok());
        REQUIRE(resized_tensor.shape() == std::vector<int64_t>({3, 1024, 1024}));
        REQUIRE(resized_tensor.dtype() == DType::FLOAT32);
        Tensor resized_image;
        imageop::image_from_tensor(resized_tensor, resized_image);
        io::save_image(
            (testing::get_output_path() / "000001_resized_upsampled.jpg").string(),
            resized_image
        );
    }
}
}  // namespace p10::tensorop
