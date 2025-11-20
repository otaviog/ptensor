#pragma once

#include <cinttypes>
#include <cstdint>

namespace p10 {
    template<typename scalar_t>
    class Acessor1D {
        public:
            Acessor1D(scalar_t *array, int64_t stride)
                : array_(array), stride_(stride) {}

            scalar_t& operator[](int64_t index) {
                // Add bounds checking if necessary
                return *(array_ + index * stride_);
            }

        private:
            scalar_t *array_;
            int64_t stride_;
    };

    class Acessor2D {
        public:
            Acessor2D(float *array, int64_t row_stride, int64_t col_stride)
                : array_(array), row_stride_(row_stride), col_stride_(col_stride) {}

            Acessor1D<float> operator[](int64_t row_index) {
                // Add bounds checking if necessary
                return Acessor1D<float>(
                    array_ + row_index * row_stride_,
                    col_stride_
                );
            }

        private:
            float *array_;
            int64_t row_stride_;
            int64_t col_stride_;
    };

    class Acessor3D {
        public:
            Acessor3D(float *array, int64_t dim0_stride, int64_t dim1_stride, int64_t dim2_stride)
                : array_(array), dim0_stride_(dim0_stride),
                  dim1_stride_(dim1_stride), dim2_stride_(dim2_stride) {}

            Acessor2D operator[](int64_t dim0_index) {
                // Add bounds checking if necessary
                return Acessor2D(
                    array_ + dim0_index * dim0_stride_,
                    dim1_stride_,
                    dim2_stride_
                );
            }

        private:
            float *array_;
            int64_t dim0_stride_;
            int64_t dim1_stride_;
            int64_t dim2_stride_;
    };
}