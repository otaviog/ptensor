#include "resize.hpp"

#include <cstddef>
#include <cstdint>

#include <immintrin.h>  // AVX2 intrinsics
#include <ptensor/dtype.hpp>
#include <ptensor/simd/compiler.hpp>
#include <ptensor/simd/cpuid.hpp>
#include <ptensor/tensor.hpp>

namespace p10::op {
namespace {
    template<typename T>
    P10Error resize_ref_impl(
        Accessor3D<const T> input,
        Accessor3D<T> output,
        int64_t new_width,
        int64_t new_height
    );

    PTENSOR_AVX2 P10Error resize_avx2_impl(
        Accessor3D<const uint8_t> input,
        Accessor3D<uint8_t> output,
        int64_t new_width,
        int64_t new_height
    );
}  // namespace

P10Error resize(const Tensor& input, Tensor& output, size_t new_width, size_t new_height) {
    return input.dtype().match([&](auto type_tag) {
        using scalar_t = typename decltype(type_tag)::type;

        auto input_accessor_res = input.as_accessor3d<scalar_t>();
        if (input_accessor_res.is_error()) {
            return input_accessor_res.error();
        }
        auto input_accessor = input_accessor_res.unwrap();

        P10_RETURN_IF_ERROR(output.create(
            make_shape(int64_t(input_accessor.channels()), int64_t(new_height), int64_t(new_width)),
            input.dtype()
        ));

        const auto default_impl = [&]() {
            return resize_ref_impl<scalar_t>(
                input_accessor,
                output.as_accessor3d<scalar_t>().unwrap(),
                int64_t(new_width),
                int64_t(new_height)
            );
        };

        if constexpr (std::is_same_v<scalar_t, uint8_t>) {
            if (simd::is_avx2_supported() && input.is_contiguous()) {
                return resize_avx2_impl(
                    input_accessor,
                    output.as_accessor3d<scalar_t>().unwrap(),
                    int64_t(new_width),
                    int64_t(new_height)
                );
            } else {
                return default_impl();
            }
        } else {
            return default_impl();
        }
    });
}

namespace {
    template<typename T>
    P10Error resize_ref_impl(
        Accessor3D<const T> input,
        Accessor3D<T> output,
        int64_t new_width,
        int64_t new_height
    ) {
        const auto channels = input.channels();
        const auto height = input.rows();
        const auto width = input.cols();

        const float x_scale = float(width) / float(new_width);
        const float y_scale = float(height) / float(new_height);

        for (int64_t chn = 0; chn < channels; ++chn) {
            auto plane_out = output[chn];
            auto plane_in = input[chn];

            for (int64_t row = 0; row < new_height; ++row) {
                const auto src_y = std::min(int64_t(float(row) * y_scale), height - 1);

                auto row_out = plane_out[row];
                auto row_in = plane_in[src_y];

                for (int64_t col = 0; col < new_width; ++col) {
                    const auto src_x = std::min(int64_t(float(col) * x_scale), width - 1);

                    row_out[col] = row_in[src_x];
                }
            }
        }
        return P10Error::Ok;
    }

    PTENSOR_AVX2 P10Error resize_avx2_impl(
        Accessor3D<const uint8_t> input,
        Accessor3D<uint8_t> output,
        int64_t new_width,
        int64_t new_height
    ) {
        const auto channels = input.channels();
        const int32_t height = int32_t(input.rows());
        const int32_t width = int32_t(input.cols());

        const float x_scale = float(width) / float(new_width);
        const float y_scale = float(height) / float(new_height);

        // Prepare constants for SIMD
        const __m256 x_scale_vec = _mm256_set1_ps(x_scale);
        const __m256i width_max_vec = _mm256_set1_epi32(width - 1);

        for (int64_t chn = 0; chn < channels; ++chn) {
            auto plane_out = output[chn];
            auto plane_in = input[chn];

            for (int64_t row = 0; row < new_height; ++row) {
                const auto src_y = std::min(int32_t(float(row) * y_scale), height - 1);

                auto row_out = plane_out[row];
                auto row_in = plane_in[src_y];

                int64_t col = 0;

                // Process 8 pixels at a time with SIMD
                for (; col + 8 <= new_width; col += 8) {
                    __m256i col_indices = _mm256_add_epi32(
                        _mm256_set1_epi32(static_cast<int32_t>(col)),
                        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)
                    );

                    __m256 src_x_f = _mm256_mul_ps(_mm256_cvtepi32_ps(col_indices), x_scale_vec);

                    __m256i src_x = _mm256_cvtps_epi32(src_x_f);

                    src_x = _mm256_min_epi32(src_x, width_max_vec);
                    src_x = _mm256_max_epi32(src_x, _mm256_setzero_si256());

                    __m256i gathered = _mm256_i32gather_epi32(
                        reinterpret_cast<const int*>(row_in.data()),
                        src_x,
                        1
                    );

                    alignas(32) int32_t temp[8];
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(temp), gathered);

                    for (int i = 0; i < 8; ++i) {
                        row_out[col + i] = static_cast<uint8_t>(temp[i] & 0xFF);
                    }
                }

                // Handle remaining pixels with scalar code
                for (; col < new_width; ++col) {
                    const auto src_x = std::min(int32_t(float(col) * x_scale), width - 1);
                    row_out[col] = row_in[src_x];
                }
            }
        }

        return P10Error::Ok;
    }
}  // namespace

}  // namespace p10::op