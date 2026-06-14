#include <random>

#include <benchmark/benchmark.h>
#include <ptensor/tensor.hpp>

#include <p10_internal/simd/compiler.hpp>
#include <p10_internal/simd/tile2d.hpp>

#include "tensor.transpose.avx2.hpp"
#include "tensor.transpose.neon.hpp"
#include "tensor.transpose.portable.hpp"

namespace p10 {
namespace {

    // A deterministic rows x cols random tensor (fixed seed for reproducibility).
    Tensor make_input(int64_t rows, int64_t cols, Dtype dtype) {
        std::mt19937_64 const rng(42);
        return Tensor::from_random(make_shape(rows, cols), rng, TensorOptions().dtype(dtype))
            .unwrap();
    }

    // Time Tensor::transpose (tiled + cpuid-dispatched SIMD) over a rows x cols
    // tensor of element type ScalarT. Backs every square/rectangular/dtype case.
    template<typename ScalarT>
    void run_transpose(benchmark::State& state, Dtype dtype, int64_t rows, int64_t cols) {
        Tensor const input = make_input(rows, cols, dtype);
        Tensor output;

        for (auto _ : state) {
            input.transpose(output);
            benchmark::DoNotOptimize(output);
            benchmark::ClobberMemory();
        }

        const int64_t elements = rows * cols;
        state.SetItemsProcessed(state.iterations() * elements);
        state.SetBytesProcessed(state.iterations() * elements * sizeof(ScalarT));
    }

    void BM_Transpose_Float32(benchmark::State& state) {
        const int64_t size = state.range(0);
        run_transpose<float>(state, Dtype::Float32, size, size);
    }

    void BM_Transpose_Rectangular(benchmark::State& state) {
        run_transpose<float>(state, Dtype::Float32, state.range(0), state.range(1));
    }

    void BM_Transpose_Int32(benchmark::State& state) {
        const int64_t size = state.range(0);
        run_transpose<int32_t>(state, Dtype::Int32, size, size);
    }

    // Baseline: textbook nested-loop transpose, no tiling and no SIMD. Used to
    // measure what Tensor::transpose (tiled + SIMD 8x8) buys over the naive path.
    void BM_Transpose_Naive_Int32(benchmark::State& state) {
        const int size = static_cast<int>(state.range(0));

        Tensor const input = make_input(size, size, Dtype::Int32);
        Tensor output;
        output.create(make_shape(size, size), TensorOptions().dtype(Dtype::Int32));

        const auto src = input.as_span2d<const int32_t>().unwrap();
        auto dst = output.as_span2d<int32_t>().unwrap();

        for (auto _ : state) {
            for (int64_t i = 0; i < src.height(); ++i) {
                for (int64_t j = 0; j < src.width(); ++j) {
                    dst.row(j)[i] = src.row(i)[j];
                }
            }
            benchmark::DoNotOptimize(output);
            benchmark::ClobberMemory();
        }

        const int64_t elements = static_cast<int64_t>(size) * size;
        state.SetItemsProcessed(state.iterations() * elements);
        state.SetBytesProcessed(state.iterations() * elements * sizeof(int32_t));
    }

    // Time one specific transpose kernel directly, bypassing the cpuid dispatch
    // in Tensor::transpose. make_kernel builds the SIMD spec under test from the
    // tile closures; tile2d_autoblock drives it with the scalar border kernel.
    template<typename MakeKernel>
    void run_kernel_int32(benchmark::State& state, MakeKernel make_kernel) {
        constexpr size_t SIMD_BLOCK = 8;
        const int size = static_cast<int>(state.range(0));

        std::mt19937_64 const rng(42);
        Tensor const input =
            Tensor::from_random(make_shape(size, size), rng, TensorOptions().dtype(Dtype::Int32))
                .unwrap();
        Tensor output;
        output.create(make_shape(size, size), TensorOptions().dtype(Dtype::Int32));

        const auto src = input.as_span2d<const int32_t>().unwrap();
        auto dst = output.as_span2d<int32_t>().unwrap();
        const int64_t src_stride = src.width();
        const int64_t dst_stride = dst.width();

        const auto src_block = [&](const TileRegion2D& region) {
            return &src.row(region.row)[region.col];
        };
        const auto dst_block = [&](const TileRegion2D& region) {
            return &dst.row(region.col)[region.row];
        };

        auto kernel = make_kernel(src_block, dst_block, src_stride, dst_stride);
        auto border =
            make_transpose_border<int32_t>(src_block, dst_block, src_stride, dst_stride);

        for (auto _ : state) {
            simd::tile2d_autoblock<SIMD_BLOCK, int32_t>(
                src.height(), src.width(), kernel.fn, border
            );
            benchmark::DoNotOptimize(dst);
            benchmark::ClobberMemory();
        }

        const int64_t elements = static_cast<int64_t>(size) * size;
        state.SetItemsProcessed(state.iterations() * elements);
        state.SetBytesProcessed(state.iterations() * elements * sizeof(int32_t));
    }

    void BM_Kernel_Portable(benchmark::State& state) {
        run_kernel_int32(state, [](auto sb, auto db, int64_t ss, int64_t ds) {
            return make_portable_transpose<8, int32_t>(sb, db, ss, ds);
        });
    }

    // Element-by-element transpose, but driven through tile2d_autoblock so it is
    // cache-blocked. Compare against BM_Transpose_Naive_Int32 (same work, no
    // tiling) to isolate the blocking win from the SIMD win.
    void BM_Kernel_Scalar(benchmark::State& state) {
        run_kernel_int32(state, [](auto sb, auto db, int64_t ss, int64_t ds) {
            return simd::Portable<8>(make_transpose_border<int32_t>(sb, db, ss, ds));
        });
    }

#if PTENSOR_HAS_INTRINSICS_H
    void BM_Kernel_Avx2(benchmark::State& state) {
        run_kernel_int32(state, [](auto sb, auto db, int64_t ss, int64_t ds) {
            return make_avx2_transpose<8>(sb, db, ss, ds);
        });
    }
#endif

#if PTENSOR_HAS_NEON
    void BM_Kernel_Neon(benchmark::State& state) {
        run_kernel_int32(state, [](auto sb, auto db, int64_t ss, int64_t ds) {
            return make_neon_transpose<8>(sb, db, ss, ds);
        });
    }
#endif

    // Per-kernel comparison: only kernels the target can actually emit are
    // registered (empty stand-in kernels are never benchmarked).
    BENCHMARK(BM_Kernel_Scalar)->Arg(256)->Arg(1024)->Arg(2048)->Unit(benchmark::kMicrosecond);
    BENCHMARK(BM_Kernel_Portable)->Arg(256)->Arg(1024)->Arg(2048)->Unit(benchmark::kMicrosecond);
#if PTENSOR_HAS_INTRINSICS_H
    BENCHMARK(BM_Kernel_Avx2)->Arg(256)->Arg(1024)->Arg(2048)->Unit(benchmark::kMicrosecond);
#endif
#if PTENSOR_HAS_NEON
    BENCHMARK(BM_Kernel_Neon)->Arg(256)->Arg(1024)->Arg(2048)->Unit(benchmark::kMicrosecond);
#endif

    // Square matrices across scales: small (no cache blocking) through large
    // (blocking critical).
    BENCHMARK(BM_Transpose_Float32)
        ->Arg(4)
        ->Arg(8)
        ->Arg(16)
        ->Arg(32)
        ->Arg(64)
        ->Arg(128)
        ->Arg(256)
        ->Arg(512)
        ->Arg(1024)
        ->Arg(2048)
        ->Arg(4096)
        ->Unit(benchmark::kMicrosecond);

    // Rectangular matrices
    BENCHMARK(BM_Transpose_Rectangular)
        ->Args({100, 1000})
        ->Args({1000, 100})
        ->Args({512, 2048})
        ->Args({2048, 512})
        ->Unit(benchmark::kMicrosecond);

    // Test SIMD path with int32
    BENCHMARK(BM_Transpose_Int32)
        ->Arg(64)
        ->Arg(256)
        ->Arg(1024)
        ->Arg(2048)
        ->Unit(benchmark::kMicrosecond);

    // Naive baseline, same sizes/dtype, to compare against the SIMD path above.
    BENCHMARK(BM_Transpose_Naive_Int32)
        ->Arg(64)
        ->Arg(256)
        ->Arg(1024)
        ->Arg(2048)
        ->Unit(benchmark::kMicrosecond);

}  // namespace
}  // namespace p10

BENCHMARK_MAIN();
