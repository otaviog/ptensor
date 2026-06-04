#include "crop.hpp"

#include <ptensor/dtype.hpp>
#include <ptensor/tensor.hpp>

namespace p10::op {

P10Error crop(const Tensor& image, size_t x, size_t y, size_t w, size_t h, Tensor& crop) {
    const auto image_shape = image.shape().as_span();
    if (image_shape.size() != 3) {
        return P10Error::InvalidArgument << "crop only supports 3d tensor";
    }
    if (x >= static_cast<size_t>(image_shape[2]) || y >= static_cast<size_t>(image_shape[1])) {
        return P10Error::InvalidArgument << "crop start position is out of bounds";
    }
    if (x + w > static_cast<size_t>(image_shape[2])
        || y + h > static_cast<size_t>(image_shape[1])) {
        return P10Error::InvalidArgument << "crop end position is out of bounds";
    }

    const int64_t num_channels = image_shape[0];

    P10_RETURN_IF_ERROR(crop.create(make_shape(num_channels, int64_t(h), int64_t(w)), image.dtype())
    );

    if (image.is_contiguous()) {
        return image.dtype().match([&](auto type_tag) -> P10Error {
            using scalar_t = decltype(type_tag)::type;

            auto src_res = image.as_planar_span3d<const scalar_t>();
            if (src_res.is_error()) {
                return src_res.error();
            }
            auto src = src_res.unwrap();
            auto dst = crop.as_planar_span3d<scalar_t>().unwrap();

            for (int64_t c = 0; c < num_channels; ++c) {
                auto plane_src = src[c];
                auto plane_dst = dst[c];
                for (int64_t row = 0; row < int64_t(h); ++row) {
                    auto row_src = plane_src[int64_t(y) + row].subspan(int64_t(x), int64_t(w));
                    auto row_dst = plane_dst[row];
                    std::copy(row_src.begin(), row_src.end(), row_dst.begin());
                }
            }
            return P10Error::Ok;
        });
    }
    return image.dtype().match([&](auto type_tag) -> P10Error {
        using scalar_t = decltype(type_tag)::type;

        auto src_res = image.as_accessor3d<const scalar_t>();
        if (src_res.is_error()) {
            return src_res.error();
        }
        auto src = src_res.unwrap();
        auto dst = crop.as_accessor3d<scalar_t>().unwrap();

        for (int64_t c = 0; c < num_channels; ++c) {
            auto plane_src = src[c];
            auto plane_dst = dst[c];
            for (int64_t row = 0; row < int64_t(h); ++row) {
                auto row_src = plane_src[int64_t(y) + row];
                auto row_dst = plane_dst[row];
                for (int64_t col = 0; col < int64_t(w); ++col) {
                    row_dst[col] = row_src[int64_t(x) + col];
                }
            }
        }

        return P10Error::Ok;
    });
}

}  // namespace p10::op
