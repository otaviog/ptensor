#include <random>

#include <benchmark/benchmark.h>
#include <ptensor/op/blur.hpp>
#include <ptensor/tensor.hpp>

namespace p10::op {
namespace {
    // NOLINTNEXTLINE(readability-identifier-naming) -- BM_ is the Google Benchmark convention.
    void BM_Blur(benchmark::State& state) {
        const int width = static_cast<int>(state.range(0));
        const int height = static_cast<int>(state.range(1));
        const int kernel_size = static_cast<int>(state.range(2));

        const Tensor input =
            Tensor::from_random(make_shape(height, width), std::mt19937_64(std::random_device {}()))
                .unwrap();

        GaussianBlur blur_op = GaussianBlur::create(kernel_size, 1.0).unwrap();

        Tensor output;
        output.create_like(input);
        for ([[maybe_unused]] auto _ : state) {
            blur_op.transform(input, output);
            benchmark::DoNotOptimize(output);
        }
    }

    BENCHMARK(BM_Blur)
        ->Args({256, 256, 3})
        ->Args({512, 512, 5})
        ->Args({1024, 1024, 7})
        ->Unit(benchmark::kMicrosecond);
}  // namespace
}  // namespace p10::op

BENCHMARK_MAIN();
