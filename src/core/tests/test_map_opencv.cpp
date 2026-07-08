#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <ptensor/map/opencv2.hpp>
#include <ptensor/tensor.hpp>
#include <ptensor/testing/catch2_assertions.hpp>

namespace p10 {
namespace detail {
    TEST_CASE("core::map::opencv depth conversion round trips", "[map][opencv]") {
        auto dtype = GENERATE(
            Dtype(Dtype::Uint8),
            Dtype(Dtype::Int8),
            Dtype(Dtype::Uint16),
            Dtype(Dtype::Int16),
            Dtype(Dtype::Int32),
            Dtype(Dtype::Float32),
            Dtype(Dtype::Float64)
        );
        DYNAMIC_SECTION("Testing with dtype " << to_string(dtype.value)) {
            auto depth = to_opencv_depth(dtype).unwrap();
            REQUIRE(from_opencv_depth(depth).unwrap() == dtype);
        }
    }

    TEST_CASE("core::map::opencv depth conversion rejects unsupported types", "[map][opencv]") {
        REQUIRE_THAT(to_opencv_depth(Dtype::Uint32), testing::is_error(P10Error::InvalidArgument));
        REQUIRE_THAT(to_opencv_depth(Dtype::Int64), testing::is_error(P10Error::InvalidArgument));
        REQUIRE_THAT(from_opencv_depth(-1), testing::is_error(P10Error::InvalidArgument));
    }
}  // namespace detail

TEST_CASE("core::map::from_opencv copies a 3-channel mat", "[map][opencv]") {
    cv::Mat mat(4, 6, CV_8UC3);
    for (int row = 0; row < mat.rows; ++row) {
        for (int col = 0; col < mat.cols; ++col) {
            mat.at<cv::Vec3b>(row, col) = cv::Vec3b(
                static_cast<uchar>(row),
                static_cast<uchar>(col),
                static_cast<uchar>(row + col)
            );
        }
    }

    Tensor tensor;
    REQUIRE_THAT(from_opencv(mat, tensor), testing::is_ok());
    REQUIRE(tensor.dtype() == Dtype::Uint8);
    REQUIRE(tensor.usage() == Usage::Image);
    REQUIRE(tensor.dims() == 3);
    REQUIRE(tensor.shape(0).unwrap() == 4);
    REQUIRE(tensor.shape(1).unwrap() == 6);
    REQUIRE(tensor.shape(2).unwrap() == 3);

    auto accessor = tensor.as_accessor3d<uint8_t>().unwrap();
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 6; ++col) {
            REQUIRE(accessor[row][col][0] == row);
            REQUIRE(accessor[row][col][1] == col);
            REQUIRE(accessor[row][col][2] == row + col);
        }
    }
}

TEST_CASE("core::map::from_opencv copies a single-channel mat as [H, W]", "[map][opencv]") {
    cv::Mat mat(3, 5, CV_32F);
    for (int row = 0; row < mat.rows; ++row) {
        for (int col = 0; col < mat.cols; ++col) {
            mat.at<float>(row, col) = static_cast<float>(row * 5 + col);
        }
    }

    Tensor tensor;
    REQUIRE_THAT(from_opencv(mat, tensor), testing::is_ok());
    REQUIRE(tensor.dims() == 2);
    REQUIRE(tensor.dtype() == Dtype::Float32);

    auto data = tensor.as_span1d<float>().unwrap();
    for (int i = 0; i < 15; ++i) {
        REQUIRE(data[i] == static_cast<float>(i));
    }
}

TEST_CASE("core::map::from_opencv copies a non-continuous ROI", "[map][opencv]") {
    cv::Mat full(8, 8, CV_8UC1);
    for (int row = 0; row < 8; ++row) {
        for (int col = 0; col < 8; ++col) {
            full.at<uchar>(row, col) = static_cast<uchar>(row * 8 + col);
        }
    }
    cv::Mat roi = full(cv::Rect(2, 1, 4, 3));
    REQUIRE(!roi.isContinuous());

    Tensor tensor;
    REQUIRE_THAT(from_opencv(roi, tensor), testing::is_ok());
    REQUIRE(tensor.shape(0).unwrap() == 3);
    REQUIRE(tensor.shape(1).unwrap() == 4);

    auto accessor = tensor.as_accessor2d<uint8_t>().unwrap();
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
            REQUIRE(accessor[row][col] == (row + 1) * 8 + (col + 2));
        }
    }
}

TEST_CASE("core::map::from_opencv_view shares data with the mat", "[map][opencv]") {
    cv::Mat mat(4, 4, CV_8UC1, cv::Scalar(0));

    auto tensor = from_opencv_view(mat).unwrap();
    REQUIRE(tensor.dims() == 2);
    REQUIRE(tensor.is_contiguous());

    mat.at<uchar>(2, 3) = 77;
    auto accessor = tensor.as_accessor2d<uint8_t>().unwrap();
    REQUIRE(accessor[2][3] == 77);

    accessor[0][1] = 55;
    REQUIRE(mat.at<uchar>(0, 1) == 55);
}

TEST_CASE("core::map::from_opencv_view of an ROI keeps the row stride", "[map][opencv]") {
    cv::Mat full(6, 6, CV_8UC1);
    for (int row = 0; row < 6; ++row) {
        for (int col = 0; col < 6; ++col) {
            full.at<uchar>(row, col) = static_cast<uchar>(row * 6 + col);
        }
    }
    cv::Mat roi = full(cv::Rect(1, 2, 3, 2));

    auto tensor = from_opencv_view(roi).unwrap();
    REQUIRE(tensor.stride(0).unwrap() == 6);
    REQUIRE(!tensor.is_contiguous());

    auto accessor = tensor.as_accessor2d<uint8_t>().unwrap();
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 3; ++col) {
            REQUIRE(accessor[row][col] == (row + 2) * 6 + (col + 1));
        }
    }
}

TEST_CASE("core::map::to_opencv copies a tensor into a mat", "[map][opencv]") {
    auto tensor = Tensor::from_range(make_shape(2, 3, 3), TensorOptions(Dtype::Uint8)).unwrap();

    cv::Mat mat;
    REQUIRE_THAT(to_opencv(tensor, mat), testing::is_ok());
    REQUIRE(mat.rows == 2);
    REQUIRE(mat.cols == 3);
    REQUIRE(mat.type() == CV_8UC3);

    int expected = 0;
    for (int row = 0; row < mat.rows; ++row) {
        for (int col = 0; col < mat.cols; ++col) {
            const auto& pixel = mat.at<cv::Vec3b>(row, col);
            for (int channel = 0; channel < 3; ++channel) {
                REQUIRE(pixel[channel] == expected++);
            }
        }
    }
}

TEST_CASE("core::map::to_opencv rejects unsupported ranks", "[map][opencv]") {
    auto tensor = Tensor::from_range(make_shape(2, 2, 2, 2)).unwrap();
    cv::Mat mat;
    REQUIRE_THAT(to_opencv(tensor, mat), testing::is_error(P10Error::InvalidArgument));
}

TEST_CASE("core::map::to_opencv_view shares data with the tensor", "[map][opencv]") {
    auto tensor = Tensor::from_range(make_shape(3, 4), TensorOptions(Dtype::Uint8)).unwrap();

    auto mat = to_opencv_view(tensor).unwrap();
    REQUIRE(mat.rows == 3);
    REQUIRE(mat.cols == 4);
    REQUIRE(mat.type() == CV_8UC1);
    REQUIRE(mat.at<uchar>(1, 1) == 5);

    mat.at<uchar>(2, 2) = 99;
    auto accessor = tensor.as_accessor2d<uint8_t>().unwrap();
    REQUIRE(accessor[2][2] == 99);
}

TEST_CASE("core::map::opencv round trip preserves values", "[map][opencv]") {
    auto source = Tensor::from_range(make_shape(5, 4, 3), TensorOptions(Dtype::Uint8)).unwrap();

    cv::Mat mat;
    REQUIRE_THAT(to_opencv(source, mat), testing::is_ok());
    Tensor round_trip;
    REQUIRE_THAT(from_opencv(mat, round_trip), testing::is_ok());

    auto expected = source.as_span1d<uint8_t>().unwrap();
    auto actual = round_trip.as_span1d<uint8_t>().unwrap();
    REQUIRE(expected.size() == actual.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(expected[i] == actual[i]);
    }
}

}  // namespace p10
