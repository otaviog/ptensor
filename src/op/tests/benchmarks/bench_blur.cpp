#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include <ptensor/op/blur.hpp>
#include <ptensor/tensor.hpp>

// Private kernel header: the per-kernel benchmarks call the convolution sweeps
// directly, bypassing GaussianBlur::transform and the tile2d dispatch.
#include "blur.hblur.hpp"

namespace p10::op {
namespace {

    // A normalized box kernel; the weights do not affect the timing, only the
    // tap count (2*half + 1) does.
    std::vector<float> make_kernel(int size) {
        return std::vector<float>(static_cast<size_t>(size), 1.0F / static_cast<float>(size));
    }

    Tensor random_plane(int height, int width) {
        return Tensor::from_random(
                   make_shape(height, width),
                   std::mt19937_64(42),
                   TensorOptions().dtype(Dtype::Float32)
        )
            .unwrap();
    }

    // Whole operator: both separable passes, tiling, kernel dispatch and the
    // intermediate transpose. The end-to-end number to compare against.
    // NOLINTNEXTLINE(readability-identifier-naming) -- BM_ is the Google Benchmark convention.
    void BM_Blur(benchmark::State& state) {
        const int height = static_cast<int>(state.range(0));
        const int width = static_cast<int>(state.range(1));
        const int kernel_size = static_cast<int>(state.range(2));

        const Tensor input = random_plane(height, width);
        GaussianBlur blur_op = GaussianBlur::create(kernel_size, 1.0).unwrap();

        Tensor output;
        output.create_like(input);
        for ([[maybe_unused]] auto _ : state) {
            blur_op.transform(input, output);
            benchmark::DoNotOptimize(output);
        }
        state.SetItemsProcessed(state.iterations() * height * width);
    }

    // Interior kernel only: no edge clamp, KHALF a compile-time constant so the
    // tap loop unrolls exactly as the real interior path. Swept over every plane
    // row directly (no tiler), over the halo-inset columns the interior covers.
    template<int64_t KHALF>
    void run_portable(benchmark::State& state) {
        const int height = static_cast<int>(state.range(0));
        const int width = static_cast<int>(state.range(1));
        const auto kernel = make_kernel(2 * KHALF + 1);

        const Tensor input = random_plane(height, width);
        Tensor output;
        output.create_like(input);
        const auto in = input.as_span2d<const float>().unwrap();
        auto out = output.as_span2d<float>().unwrap();
        const int64_t inner = width - 2 * KHALF;

        for ([[maybe_unused]] auto _ : state) {
            for (int64_t row = 0; row < height; ++row) {
                hblur_portable<float, KHALF>(
                    in(row).slice(KHALF, inner),
                    out(row).slice(KHALF, inner),
                    kernel.data()
                );
            }
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * height * inner);
    }

    // Border kernel only: clamps every tap (runtime half), swept over every full
    // row. Compare against run_portable to see the clamp cost.
    void run_scalar(benchmark::State& state) {
        const int height = static_cast<int>(state.range(0));
        const int width = static_cast<int>(state.range(1));
        const int half = static_cast<int>(state.range(2));
        const auto kernel = make_kernel(2 * half + 1);

        const Tensor input = random_plane(height, width);
        Tensor output;
        output.create_like(input);
        const auto in = input.as_span2d<const float>().unwrap();
        auto out = output.as_span2d<float>().unwrap();

        for ([[maybe_unused]] auto _ : state) {
            for (int64_t row = 0; row < height; ++row) {
                hblur_scalar<float>(in(row), out(row), half, kernel.data());
            }
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * height * width);
    }

    BENCHMARK(BM_Blur)
        ->Args({256, 256, 3})
        ->Args({512, 512, 5})
        ->Args({1024, 1024, 7})
        ->Unit(benchmark::kMicrosecond);

    BENCHMARK_TEMPLATE(run_portable, 3)
        ->Args({256, 256})
        ->Args({512, 512})
        ->Args({1024, 1024})
        ->Unit(benchmark::kMicrosecond);

    BENCHMARK(run_scalar)
        ->Args({256, 256, 3})
        ->Args({512, 512, 3})
        ->Args({1024, 1024, 3})
        ->Unit(benchmark::kMicrosecond);
}  // namespace
}  // namespace p10::op

BENCHMARK_MAIN();
