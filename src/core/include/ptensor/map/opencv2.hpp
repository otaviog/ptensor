#pragma once

// Header-only OpenCV adapters. Compiles to nothing when OpenCV is not on the
// include path; check P10_MAP_HAS_OPENCV to know if the functions exist.
#if __has_include(<opencv2/core.hpp>)
    #define P10_MAP_HAS_OPENCV 1

    #include <cstring>

    #include <opencv2/core.hpp>

    #include "../p10_error.hpp"
    #include "../p10_result.hpp"
    #include "../tensor.hpp"

namespace p10 {

// Helper conversions (kept in namespace p10 so unit tests can call them)
/// Converts a dtype to an OpenCV depth (CV_8U, CV_32F, ...).
inline P10Result<int> to_opencv_depth(Dtype dtype) {
    switch (dtype.value) {
        case Dtype::Uint8:
            return Ok(CV_8U);
        case Dtype::Int8:
            return Ok(CV_8S);
        case Dtype::Uint16:
            return Ok(CV_16U);
        case Dtype::Int16:
            return Ok(CV_16S);
        case Dtype::Int32:
            return Ok(CV_32S);
        case Dtype::Float16:
            return Ok(CV_16F);
        case Dtype::Float32:
            return Ok(CV_32F);
        case Dtype::Float64:
            return Ok(CV_64F);
        default:
            return Err(P10Error::InvalidArgument << "Dtype has no OpenCV depth equivalent");
    }
}

/// Converts an OpenCV depth (CV_8U, CV_32F, ...) to a dtype.
inline P10Result<Dtype> from_opencv_depth(int depth) {
    switch (depth) {
        case CV_8U:
            return Ok(Dtype(Dtype::Uint8));
        case CV_8S:
            return Ok(Dtype(Dtype::Int8));
        case CV_16U:
            return Ok(Dtype(Dtype::Uint16));
        case CV_16S:
            return Ok(Dtype(Dtype::Int16));
        case CV_32S:
            return Ok(Dtype(Dtype::Int32));
        case CV_16F:
            return Ok(Dtype(Dtype::Float16));
        case CV_32F:
            return Ok(Dtype(Dtype::Float32));
        case CV_64F:
            return Ok(Dtype(Dtype::Float64));
        default:
            return Err(P10Error::InvalidArgument << "OpenCV depth has no dtype equivalent");
    }
}
}

/// Copies a 2D `cv::Mat` into `tensor`, allocating it as `[H, W]`
/// (single channel) or `[H, W, C]` with `Usage::Image`. Handles
/// non-continuous mats (ROIs).
inline P10Error from_opencv(const cv::Mat& mat, Tensor& tensor) {
    if (mat.empty()) {
        return P10Error::InvalidArgument << "Mat is empty";
    }
    if (mat.dims != 2) {
        return P10Error::NotImplemented << "Only 2D mats are supported";
    }
    auto dtype_res = from_opencv_depth(mat.depth());
    if (dtype_res.is_error()) {
        return dtype_res.error();
    }

    const int64_t channels = mat.channels();
    const Shape shape =
        channels == 1 ? make_shape(mat.rows, mat.cols) : make_shape(mat.rows, mat.cols, channels);
    P10_RETURN_IF_ERROR(
        tensor.create(shape, TensorOptions(dtype_res.unwrap()).usage(Usage::Image))
    );

    const size_t row_bytes = static_cast<size_t>(mat.cols) * mat.elemSize();
    auto dst = tensor.as_bytes();
    for (int row = 0; row < mat.rows; ++row) {
        std::memcpy(dst.data() + static_cast<size_t>(row) * row_bytes, mat.ptr(row), row_bytes);
    }
    return P10Error::Ok;
}

/// Returns a tensor view of a 2D `cv::Mat` without copying. Non-continuous
/// mats (ROIs) are represented with a row stride.
///
/// The tensor does not hold a reference on the mat: the mat's data must
/// outlive the returned tensor.
inline P10Result<Tensor> from_opencv_view(cv::Mat& mat) {
    if (mat.empty()) {
        return Err(P10Error::InvalidArgument << "Mat is empty");
    }
    if (mat.dims != 2) {
        return Err(P10Error::NotImplemented << "Only 2D mats are supported");
    }
    auto dtype_res = from_opencv_depth(mat.depth());
    if (dtype_res.is_error()) {
        return Err(dtype_res.error());
    }

    const int64_t channels = mat.channels();
    const auto row_stride = static_cast<int64_t>(mat.step1(0));
    const Shape shape =
        channels == 1 ? make_shape(mat.rows, mat.cols) : make_shape(mat.rows, mat.cols, channels);
    const Stride stride =
        channels == 1 ? make_stride(row_stride, 1) : make_stride(row_stride, channels, 1);
    return Ok(
        Tensor::from_data(
            static_cast<void*>(mat.data),
            shape,
            TensorOptions(dtype_res.unwrap()).stride(stride).usage(Usage::Image)
        )
    );
}

/// Copies a `[H, W]` or `[H, W, C]` tensor into `mat`, allocating it if
/// needed.
inline P10Error to_opencv(const Tensor& tensor, cv::Mat& mat) {
    if (tensor.empty()) {
        return P10Error::InvalidArgument << "Tensor is empty";
    }
    if (tensor.device() != Device::Cpu) {
        return P10Error::NotImplemented << "OpenCV adapters require CPU tensors";
    }
    if (tensor.dims() != 2 && tensor.dims() != 3) {
        return P10Error::InvalidArgument << "Tensor must be [H, W] or [H, W, C]";
    }

    const auto shape = tensor.shape().as_span();
    const int64_t channels = tensor.dims() == 3 ? shape[2] : 1;
    if (channels > CV_CN_MAX) {
        return P10Error::InvalidArgument << "Tensor has too many channels for OpenCV";
    }
    auto depth_res = to_opencv_depth(tensor.dtype());
    if (depth_res.is_error()) {
        return depth_res.error();
    }

    Tensor contiguous;
    const Tensor* src = &tensor;
    if (!tensor.is_contiguous()) {
        auto contiguous_res = tensor.to_contiguous();
        if (contiguous_res.is_error()) {
            return contiguous_res.error();
        }
        contiguous = contiguous_res.unwrap();
        src = &contiguous;
    }

    mat.create(
        static_cast<int>(shape[0]),
        static_cast<int>(shape[1]),
        CV_MAKETYPE(depth_res.unwrap(), static_cast<int>(channels))
    );
    const auto bytes = src->as_bytes();
    if (mat.isContinuous()) {
        std::memcpy(mat.data, bytes.data(), bytes.size());
    } else {
        const size_t row_bytes = static_cast<size_t>(mat.cols) * mat.elemSize();
        for (int row = 0; row < mat.rows; ++row) {
            std::memcpy(
                mat.ptr(row),
                bytes.data() + static_cast<size_t>(row) * row_bytes,
                row_bytes
            );
        }
    }
    return P10Error::Ok;
}

/// Returns a `cv::Mat` header over the data of a `[H, W]` or `[H, W, C]`
/// tensor without copying. Rows may be strided, but pixels must be packed
/// (innermost stride 1; for `[H, W, C]`, the W stride must equal C).
///
/// The mat does not own the data: the tensor must outlive the returned mat.
inline P10Result<cv::Mat> to_opencv_view(Tensor& tensor) {
    if (tensor.empty()) {
        return Err(P10Error::InvalidArgument << "Tensor is empty");
    }
    if (tensor.device() != Device::Cpu) {
        return Err(P10Error::NotImplemented << "OpenCV adapters require CPU tensors");
    }
    if (tensor.dims() != 2 && tensor.dims() != 3) {
        return Err(P10Error::InvalidArgument << "Tensor must be [H, W] or [H, W, C]");
    }

    const auto shape = tensor.shape().as_span();
    const auto stride = tensor.stride().as_span();
    const int64_t channels = tensor.dims() == 3 ? shape[2] : 1;
    if (channels > CV_CN_MAX) {
        return Err(P10Error::InvalidArgument << "Tensor has too many channels for OpenCV");
    }
    const bool packed_pixels =
        tensor.dims() == 2 ? stride[1] == 1 : stride[2] == 1 && stride[1] == channels;
    if (!packed_pixels) {
        return Err(P10Error::InvalidArgument << "Tensor pixels must be packed for a Mat view");
    }
    auto depth_res = to_opencv_depth(tensor.dtype());
    if (depth_res.is_error()) {
        return Err(depth_res.error());
    }

    const size_t step_bytes = static_cast<size_t>(stride[0]) * tensor.dtype().size_bytes();
    return Ok(
        cv::Mat(
            static_cast<int>(shape[0]),
            static_cast<int>(shape[1]),
            CV_MAKETYPE(depth_res.unwrap(), static_cast<int>(channels)),
            tensor.as_bytes().data(),
            step_bytes
        )
    );
}

}  // namespace p10

#endif  // __has_include(<opencv2/core.hpp>)
