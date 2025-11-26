#include <benchmark/benchmark.h>
#include <ptensor/op/blur.hpp>
#include <ptensor/tensor.hpp>

namespace p10::op {
namespace {
    static void BM_Blur(benchmark::State& state) {
        const int width = state.range(0);
        const int height = state.range(1);
        const int kernel_size = state.range(2);

        Tensor input =
            Tensor::from_random(make_shape(height, width), std::mt19937_64(std::random_device {}()))
                .unwrap();

        GaussianBlur blur_op = GaussianBlur::create(kernel_size, 1.0).unwrap();

        Tensor output;
        output.create_like(input);
        for (auto _ : state) {
            blur_op.transform(input, output);
            benchmark::DoNotOptimize(output);
        }
    }
}  // namespace
}  // namespace p10::op
