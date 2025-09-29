#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <ptensor/shape.hpp>
#include <ptensor/tensor.hpp>
#include <ptensor/tensor_options.hpp>

namespace p10 {
TEST_CASE("Tensor::full", "[tensor]") {
    auto tensor = Tensor::full(make_shape({2, 3}).unwrap(), 3.0).expect("Could not create tensor");

    REQUIRE(tensor.dtype() == Dtype::Float32);
    REQUIRE(tensor.shape().count() == 6);
    REQUIRE(tensor.stride().dims() == 2);
    CHECK(tensor.axes().dims() == 2);
    REQUIRE(tensor.dims() == 2);

    auto data = tensor.as_span1d<float>();
    for (size_t i = 0; i < tensor.shape().count(); i++) {
        REQUIRE(data[i] == 3.0);
    }
}

TEST_CASE("Tensor::zeros", "[tensor]") {
    auto tensor = Tensor::zeros(make_shape({2, 3}).unwrap()).expect("Could not create tensor");

    REQUIRE(tensor.dtype() == Dtype::Float32);
    REQUIRE(tensor.shape().count() == 6);
    REQUIRE(tensor.stride().dims() == 2);
    REQUIRE(tensor.axes().dims() == 2);
    REQUIRE(tensor.dims() == 2);

    auto data = tensor.as_span1d<float>();
    for (size_t i = 0; i < tensor.shape().count(); i++) {
        REQUIRE(data[i] == 0.0);
    }
}

TEST_CASE("Test::empty", "[tensor]") {
    auto tensor = Tensor::empty(make_shape({2, 3}).unwrap()).expect("Could not create tensor");

    REQUIRE(tensor.dtype() == Dtype::Float32);
    REQUIRE(tensor.shape().count() == 6);
    REQUIRE(tensor.stride().dims() == 2);
    REQUIRE(tensor.axes().dims() == 2);
    REQUIRE(tensor.dims() == 2);
}

TEST_CASE("Tensor::type and device", "[tensor]") {
    const auto all_dtypes = {
        Dtype::Uint8,
        Dtype::Uint16,
        Dtype::Uint32,
        Dtype::Int8,
        Dtype::Int16,
        Dtype::Int32,
        Dtype::Int64,
        Dtype::Float16,
        Dtype::Float32,
        Dtype::Float64
    };

    const auto all_devices = {Device::Cpu, Device::Cuda};

    for (const auto& dtype : all_dtypes) {
        for (const auto& device : all_devices) {
            auto tensor = Tensor::from_data(
                nullptr,
                make_shape({2, 3}).unwrap(),
                TensorOptions().dtype(dtype).device(device)
            );

            REQUIRE(tensor.dtype() == dtype);
            REQUIRE(tensor.device() == device);
        }
    }
}

TEST_CASE("Tensor::strides", "[tensor]") {
    auto tensor = Tensor::empty(make_shape({2, 3}).unwrap()).expect("Could not create tensor");

    REQUIRE(tensor.stride().dims() == 2);
    REQUIRE(tensor.stride(0).unwrap() == 3);
    REQUIRE(tensor.stride(1).unwrap() == 1);

    auto tensor2 = Tensor::empty(
                       make_shape({2, 3}).unwrap(),
                       TensorOptions().stride(make_stride({1, 2}).unwrap())
    )
                       .expect("Could not create tensor");

    REQUIRE(tensor2.stride().dims() == 2);
    REQUIRE(tensor2.stride(0).unwrap() == 1);
    REQUIRE(tensor2.stride(1).unwrap() == 2);
}

TEST_CASE("Tensor::too many dims", "[tensor]") {
    auto tensor =
        Tensor::empty(make_shape({2, 3, 4, 5}).unwrap()).expect("Unable to create tensor");

    REQUIRE(tensor.dims() == 4);
    REQUIRE(tensor.shape().dims() == 4);
    REQUIRE(tensor.stride().dims() == 4);
    REQUIRE(tensor.axes().dims() == 4);
}

TEST_CASE("Tensor::empty", "[tensor]") {
    auto tensor = Tensor::empty(make_shape({2, 3}).unwrap()).unwrap();
    REQUIRE(!tensor.empty());

    auto tensor2 = Tensor::empty(make_shape({0, 3}).unwrap()).unwrap();
    REQUIRE(tensor2.empty());
}

TEST_CASE("Tensor::contiguous", "[tensor]") {
    auto tensor = Tensor::empty(make_shape({2, 3}).unwrap()).unwrap();
    REQUIRE(tensor.is_contiguous());

    auto tensor2 = Tensor::empty(
                       make_shape({2, 3}).unwrap(),
                       TensorOptions().stride(make_stride({1, 2}).unwrap())
    )
                       .unwrap();
    REQUIRE(!tensor2.is_contiguous());
}

TEST_CASE("Tensor::to_contiguous", "[tensor]") {
    auto tensor = Tensor::empty(
                      make_shape({2, 3}).unwrap(),
                      TensorOptions().stride(make_stride({1, 2}).unwrap())
    )
                      .unwrap();

    // 0 1 2 3 4 5
    // view:
    // 0 2 4
    // 1 3 5
    for (size_t i = 0; i < tensor.shape().count(); i++) {
        tensor.as_span1d<float>()[i] = float(i);
    }

    auto tensor2 = tensor.to_contiguous().unwrap();
    REQUIRE(tensor2.is_contiguous());

    const std::array<float, 6> expected = {0, 2, 4, 1, 3, 5};
    for (size_t i = 0; i < tensor2.shape().count(); i++) {
        REQUIRE(tensor2.as_span1d<float>()[i] == Catch::Approx(expected[i]));
    }
}

TEST_CASE("Tensor::as_view", "[tensor]") {
    auto tensor = Tensor::zeros(make_shape({2, 3}).unwrap()).unwrap();

    auto tensor_view = tensor.as_view();
    REQUIRE(tensor_view.shape().dims() == 2);
    REQUIRE(tensor_view.stride().dims() == 2);
    REQUIRE(tensor_view.axes().dims() == 2);
    REQUIRE(tensor_view.dims() == 2);

    tensor_view.as_span1d<float>()[0] = 3.0;
    REQUIRE(tensor.as_span1d<float>()[0] == 3.0);
}
}  // namespace p10