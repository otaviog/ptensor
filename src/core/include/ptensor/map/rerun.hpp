#pragma once

// Header-only rerun SDK adapters. Compiles to nothing when rerun is not on
// the include path; check P10_MAP_HAS_RERUN to know if the functions exist.
//
// All adapters borrow the tensor's data (no copy): the tensor must stay
// alive until the returned archetype has been logged.
#if __has_include(<rerun.hpp>)
    #define P10_MAP_HAS_RERUN 1

    #include <cstdint>
    #include <vector>

    #include <rerun.hpp>

    #include "../p10_error.hpp"
    #include "../p10_result.hpp"
    #include "../tensor.hpp"

namespace p10 {

/// Converts a dtype to a rerun channel datatype.
inline P10Result<rerun::datatypes::ChannelDatatype> to_rerun_channel_datatype(Dtype dtype) {
    using ChannelDatatype = rerun::datatypes::ChannelDatatype;
    switch (dtype.value) {
        case Dtype::Uint8:
            return Ok(ChannelDatatype::U8);
        case Dtype::Uint16:
            return Ok(ChannelDatatype::U16);
        case Dtype::Uint32:
            return Ok(ChannelDatatype::U32);
        case Dtype::Int8:
            return Ok(ChannelDatatype::I8);
        case Dtype::Int16:
            return Ok(ChannelDatatype::I16);
        case Dtype::Int32:
            return Ok(ChannelDatatype::I32);
        case Dtype::Int64:
            return Ok(ChannelDatatype::I64);
        case Dtype::Float16:
            return Ok(ChannelDatatype::F16);
        case Dtype::Float32:
            return Ok(ChannelDatatype::F32);
        case Dtype::Float64:
            return Ok(ChannelDatatype::F64);
        default:
            return Err(P10Error::InvalidArgument << "Dtype has no rerun equivalent");
    }
}

/// Wraps a contiguous tensor as a `rerun::archetypes::Tensor` borrowing its
/// data. The tensor must outlive the returned archetype (log it first).
inline P10Result<rerun::archetypes::Tensor> to_rerun_tensor(const Tensor& tensor) {
    if (tensor.empty()) {
        return Err(P10Error::InvalidArgument << "Tensor is empty");
    }
    if (!tensor.is_contiguous()) {
        return Err(P10Error::InvalidArgument << "Tensor must be contiguous for rerun");
    }
    if (tensor.dtype() == Dtype::Float16) {
        return Err(P10Error::NotImplemented << "Float16 is not supported for rerun tensors");
    }

    const auto shape = tensor.shape().as_span();
    std::vector<uint64_t> dims(shape.begin(), shape.end());

    auto buffer = tensor.dtype().match([&](auto type_id) -> rerun::datatypes::TensorBuffer {
        using scalar_t = typename decltype(type_id)::type;
        const auto* data = reinterpret_cast<const scalar_t*>(tensor.as_bytes().data());
        return rerun::Collection<scalar_t>::borrow(data, tensor.size());
    });
    return Ok(rerun::archetypes::Tensor(std::move(dims), std::move(buffer)));
}

/// Wraps a contiguous `[H, W]` or `[H, W, C]` (C in {1, 3, 4}) tensor as a
/// `rerun::archetypes::Image` borrowing its data. Channels are interpreted
/// as L/RGB/RGBA. The tensor must outlive the returned archetype.
inline P10Result<rerun::archetypes::Image> to_rerun_image(const Tensor& tensor) {
    if (tensor.empty()) {
        return Err(P10Error::InvalidArgument << "Tensor is empty");
    }
    if (!tensor.is_contiguous()) {
        return Err(P10Error::InvalidArgument << "Tensor must be contiguous for rerun");
    }
    if (tensor.dims() != 2 && tensor.dims() != 3) {
        return Err(P10Error::InvalidArgument << "Tensor must be [H, W] or [H, W, C]");
    }

    const auto shape = tensor.shape().as_span();
    const int64_t channels = tensor.dims() == 3 ? shape[2] : 1;
    rerun::datatypes::ColorModel color_model;
    switch (channels) {
        case 1:
            color_model = rerun::datatypes::ColorModel::L;
            break;
        case 3:
            color_model = rerun::datatypes::ColorModel::RGB;
            break;
        case 4:
            color_model = rerun::datatypes::ColorModel::RGBA;
            break;
        default:
            return Err(P10Error::InvalidArgument << "Image tensor must have 1, 3 or 4 channels");
    }
    auto datatype_res = to_rerun_channel_datatype(tensor.dtype());
    if (datatype_res.is_error()) {
        return Err(datatype_res.error());
    }

    const auto bytes = tensor.as_bytes();
    auto buffer = rerun::Collection<uint8_t>::borrow(
        reinterpret_cast<const uint8_t*>(bytes.data()),
        bytes.size()
    );
    return Ok(
        rerun::archetypes::Image(
            std::move(buffer),
            rerun::WidthHeight(static_cast<uint32_t>(shape[1]), static_cast<uint32_t>(shape[0])),
            color_model,
            datatype_res.unwrap()
        )
    );
}

}  // namespace p10

#endif  // __has_include(<rerun.hpp>)
