#include <catch2/catch_test_macros.hpp>
#include <ptensor/map/eigen.hpp>
#include <ptensor/tensor.hpp>
#include <ptensor/testing/catch2_assertions.hpp>

namespace p10 {

TEST_CASE("core::map::to_eigen_map wraps a 2D tensor without copying", "[map][eigen]") {
    auto tensor = Tensor::from_range(make_shape(2, 3)).unwrap();

    auto map = to_eigen_map<float>(tensor).unwrap();
    REQUIRE(map.rows() == 2);
    REQUIRE(map.cols() == 3);
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 3; ++col) {
            REQUIRE(map(row, col) == static_cast<float>(row * 3 + col));
        }
    }

    map(1, 2) = 42.0f;
    REQUIRE(tensor.as_span1d<float>().unwrap()[5] == 42.0f);
}

TEST_CASE("core::map::to_eigen_map of a const tensor is read-only", "[map][eigen]") {
    const auto tensor = Tensor::from_range(make_shape(3, 2)).unwrap();

    auto map = to_eigen_map<float>(tensor).unwrap();
    static_assert(std::is_same_v<decltype(map), EigenConstMap<float>>);
    REQUIRE(map(2, 1) == 5.0f);
    REQUIRE(map.sum() == 15.0f);
}

TEST_CASE("core::map::to_eigen_map validates rank and dtype", "[map][eigen]") {
    auto tensor_3d = Tensor::from_range(make_shape(2, 2, 2)).unwrap();
    REQUIRE_THAT(to_eigen_map<float>(tensor_3d), testing::is_error(P10Error::InvalidArgument));

    auto tensor_f32 = Tensor::from_range(make_shape(2, 2)).unwrap();
    REQUIRE_THAT(to_eigen_map<double>(tensor_f32), testing::is_error(P10Error::InvalidArgument));
}

TEST_CASE("core::map::from_eigen copies a matrix into a tensor", "[map][eigen]") {
    Eigen::MatrixXd matrix(2, 3);
    matrix << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;

    Tensor tensor;
    REQUIRE_THAT(from_eigen(matrix, tensor), testing::is_ok());
    REQUIRE(tensor.dtype() == Dtype::Float64);
    REQUIRE(tensor.dims() == 2);
    REQUIRE(tensor.shape(0).unwrap() == 2);
    REQUIRE(tensor.shape(1).unwrap() == 3);

    auto data = tensor.as_span1d<double>().unwrap();
    for (int i = 0; i < 6; ++i) {
        REQUIRE(data[i] == static_cast<double>(i + 1));
    }
}

TEST_CASE("core::map::from_eigen copies an expression", "[map][eigen]") {
    Eigen::Matrix2f matrix;
    matrix << 1.0f, 2.0f, 3.0f, 4.0f;

    Tensor tensor;
    REQUIRE_THAT(from_eigen((matrix + matrix).eval(), tensor), testing::is_ok());

    auto map = to_eigen_map<float>(tensor).unwrap();
    REQUIRE(map(0, 0) == 2.0f);
    REQUIRE(map(1, 1) == 8.0f);
}

TEST_CASE("core::map::eigen round trip preserves values", "[map][eigen]") {
    auto source = Tensor::from_range(make_shape(4, 5)).unwrap();

    Eigen::MatrixXf const matrix = to_eigen_map<float>(source).unwrap();
    Tensor round_trip;
    REQUIRE_THAT(from_eigen(matrix, round_trip), testing::is_ok());

    auto expected = source.as_span1d<float>().unwrap();
    auto actual = round_trip.as_span1d<float>().unwrap();
    REQUIRE(expected.size() == actual.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(expected[i] == actual[i]);
    }
}

}  // namespace p10
