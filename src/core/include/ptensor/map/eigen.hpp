#pragma once

// Header-only Eigen adapters. Compiles to nothing when Eigen is not on the
// include path; check P10_MAP_HAS_EIGEN to know if the functions exist.
#if __has_include(<Eigen/Core>)
    #define P10_MAP_HAS_EIGEN 1

    #include <Eigen/Core>

    #include "../p10_error.hpp"
    #include "../p10_result.hpp"
    #include "../tensor.hpp"

namespace p10 {

template<typename scalar_t>
using EigenRowMatrix = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// An `Eigen::Map` over tensor data. Strided, so it works for
/// non-contiguous tensors too.
template<typename scalar_t>
using EigenMap = Eigen::
    Map<EigenRowMatrix<scalar_t>, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

template<typename scalar_t>
using EigenConstMap = Eigen::Map<
    const EigenRowMatrix<scalar_t>,
    Eigen::Unaligned,
    Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

/// Returns an `Eigen::Map` over a 2D tensor's data without copying.
/// Writing through the map writes into the tensor.
///
/// The map does not own the data: the tensor must outlive the returned map.
template<typename scalar_t>
inline P10Result<EigenMap<scalar_t>> to_eigen_map(Tensor& tensor) {
    if (tensor.device() != Device::Cpu) {
        return Err(P10Error::NotImplemented << "Eigen adapters require CPU tensors");
    }
    if (tensor.dims() != 2) {
        return Err(P10Error::InvalidArgument << "Tensor must be 2D for an Eigen matrix map");
    }
    auto data_res = tensor.as_span1d<scalar_t>();
    if (data_res.is_error()) {
        return Err(data_res.error());
    }

    const auto shape = tensor.shape().as_span();
    const auto stride = tensor.stride().as_span();
    return Ok(
        EigenMap<scalar_t>(
            data_res.unwrap().data(),
            shape[0],
            shape[1],
            Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(stride[0], stride[1])
        )
    );
}

template<typename scalar_t>
inline P10Result<EigenConstMap<scalar_t>> to_eigen_map(const Tensor& tensor) {
    if (tensor.device() != Device::Cpu) {
        return Err(P10Error::NotImplemented << "Eigen adapters require CPU tensors");
    }
    if (tensor.dims() != 2) {
        return Err(P10Error::InvalidArgument << "Tensor must be 2D for an Eigen matrix map");
    }
    auto data_res = tensor.as_span1d<scalar_t>();
    if (data_res.is_error()) {
        return Err(data_res.error());
    }

    const auto shape = tensor.shape().as_span();
    const auto stride = tensor.stride().as_span();
    return Ok(
        EigenConstMap<scalar_t>(
            data_res.unwrap().data(),
            shape[0],
            shape[1],
            Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(stride[0], stride[1])
        )
    );
}

/// Copies an Eigen matrix (or expression) into `tensor`, allocating it as
/// `[rows, cols]` with the matrix's scalar dtype.
template<typename derived_t>
inline P10Error from_eigen(const Eigen::MatrixBase<derived_t>& matrix, Tensor& tensor) {
    using Scalar = derived_t::Scalar;

    if (matrix.rows() == 0 || matrix.cols() == 0) {
        return P10Error::InvalidArgument << "Eigen matrix is empty";
    }
    P10_RETURN_IF_ERROR(tensor.create(
        make_shape(matrix.rows(), matrix.cols()),
        TensorOptions(Dtype::from<Scalar>())
    ));

    auto map_res = to_eigen_map<Scalar>(tensor);
    if (map_res.is_error()) {
        return map_res.error();
    }
    map_res.unwrap() = matrix;
    return P10Error::Ok;
}

}  // namespace p10

#endif  // __has_include(<Eigen/Core>)
