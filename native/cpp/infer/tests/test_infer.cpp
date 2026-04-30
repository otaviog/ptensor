#include <array>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <ptensor/infer/infer.hpp>
#include <ptensor/infer/infer_config.hpp>
#include <ptensor/tensor.hpp>
#include <ptensor/testing/catch2_assertions.hpp>

namespace p10::infer {

static constexpr auto kMnistModel = "native/cpp/infer/tests/data/mnist-12.onnx";

TEST_CASE("infer::IInfer::from_onnx", "[infer][ort]") {
    SECTION("errors on non-existent file") {
        REQUIRE_THAT(
            IInfer::from_onnx("non_existent_file.onnx", InferConfig()),
            testing::IsError(P10Error::InferError)
        );
    }

    SECTION("loads a valid model") {
        std::unique_ptr<IInfer> infer(
            IInfer::from_onnx(kMnistModel, InferConfig()).expect("Error while loading test onnx")
        );
        REQUIRE(infer != nullptr);
    }
}

TEST_CASE("infer::IInfer::get_input_count and get_output_count", "[infer][ort]") {
    // MNIST-12: one float32 input [1,1,28,28], one float32 output [1,10]
    std::unique_ptr<IInfer> infer(
        IInfer::from_onnx(kMnistModel, InferConfig()).expect("Error while loading test onnx")
    );

    REQUIRE(infer->get_input_count() == 1);
    REQUIRE(infer->get_output_count() == 1);
}

TEST_CASE("infer::IInfer::infer wrong tensor counts", "[infer][ort]") {
    std::unique_ptr<IInfer> infer(
        IInfer::from_onnx(kMnistModel, InferConfig()).expect("Error while loading test onnx")
    );

    SECTION("too few inputs") {
        std::array<Tensor, 0> inputs = {};
        std::array<Tensor, 1> outputs = {};
        REQUIRE_THAT(infer->infer(inputs, outputs), testing::IsError(P10Error::InvalidArgument));
    }

    SECTION("too many inputs") {
        std::array<Tensor, 2> inputs = {
            Tensor::zeros(make_shape(1, 1, 28, 28)).unwrap(),
            Tensor::zeros(make_shape(1, 1, 28, 28)).unwrap(),
        };
        std::array<Tensor, 1> outputs = {};
        REQUIRE_THAT(infer->infer(inputs, outputs), testing::IsError(P10Error::InvalidArgument));
    }

    SECTION("too few outputs") {
        std::array<Tensor, 1> inputs = {Tensor::zeros(make_shape(1, 1, 28, 28)).unwrap()};
        std::array<Tensor, 0> outputs = {};
        REQUIRE_THAT(infer->infer(inputs, outputs), testing::IsError(P10Error::InvalidArgument));
    }
}

TEST_CASE("infer::IInfer::infer output shape and dtype", "[infer][ort]") {
    // MNIST-12 maps [1,1,28,28] float32 -> [1,10] float32
    std::unique_ptr<IInfer> infer(
        IInfer::from_onnx(kMnistModel, InferConfig()).expect("Error while loading test onnx")
    );

    std::array<Tensor, 1> inputs = {Tensor::zeros(make_shape(1, 1, 28, 28)).unwrap()};
    std::array<Tensor, 1> outputs = {};

    REQUIRE_THAT(infer->infer(inputs, outputs), testing::IsOk());

    REQUIRE(outputs[0].dtype() == Dtype::Float32);
    REQUIRE(outputs[0].dims() == 2);
    REQUIRE(outputs[0].shape(0).unwrap() == 1);
    REQUIRE(outputs[0].shape(1).unwrap() == 10);
}

}  // namespace p10::infer
