#include <random>

#include <benchmark/benchmark.h>
#include <ptensor/tensor.hpp>

namespace p10 {
namespace {

    static void BM_Transpose_Small(benchmark::State& state) {
        const int size = static_cast<int>(state.range(0));

        std::mt19937_64 rng(42);
        Tensor input =
            Tensor::from_random(make_shape(size, size), rng, TensorOptions().dtype(Dtype::Float32))
                .unwrap();
        Tensor output;

        for (auto _ : state) {
            input.transpose(output);
            benchmark::DoNotOptimize(output);
            benchmark::ClobberMemory();
        }

        // Report throughput
        const int64_t elements = static_cast<int64_t>(size) * size;
        state.SetItemsProcessed(state.iterations() * elements);
        state.SetBytesProcessed(state.iterations() * elements * sizeof(float));
    }

    static void BM_Transpose_Medium(benchmark::State& state) {
        const int size = static_cast<int>(state.range(0));

        std::mt19937_64 rng(42);
        Tensor input =
            Tensor::from_random(make_shape(size, size), rng, TensorOptions().dtype(Dtype::Float32))
                .unwrap();
        Tensor output;

        for (auto _ : state) {
            input.transpose(output);
            benchmark::DoNotOptimize(output);
            benchmark::ClobberMemory();
        }

        const int64_t elements = static_cast<int64_t>(size) * size;
        state.SetItemsProcessed(state.iterations() * elements);
        state.SetBytesProcessed(state.iterations() * elements * sizeof(float));
    }

    static void BM_Transpose_Large(benchmark::State& state) {
        const int size = static_cast<int>(state.range(0));

        std::mt19937_64 rng(42);
        Tensor input =
            Tensor::from_random(make_shape(size, size), rng, TensorOptions().dtype(Dtype::Float32))
                .unwrap();
        Tensor output;

        for (auto _ : state) {
            input.transpose(output);
            benchmark::DoNotOptimize(output);
            benchmark::ClobberMemory();
        }

        const int64_t elements = static_cast<int64_t>(size) * size;
        state.SetItemsProcessed(state.iterations() * elements);
        state.SetBytesProcessed(state.iterations() * elements * sizeof(float));
    }

    static void BM_Transpose_Rectangular(benchmark::State& state) {
        const int rows = static_cast<int>(state.range(0));
        const int cols = static_cast<int>(state.range(1));

        std::mt19937_64 rng(42);
        Tensor input =
            Tensor::from_random(make_shape(rows, cols), rng, TensorOptions().dtype(Dtype::Float32))
                .unwrap();
        Tensor output;

        for (auto _ : state) {
            input.transpose(output);
            benchmark::DoNotOptimize(output);
            benchmark::ClobberMemory();
        }

        const int64_t elements = static_cast<int64_t>(rows) * cols;
        state.SetItemsProcessed(state.iterations() * elements);
        state.SetBytesProcessed(state.iterations() * elements * sizeof(float));
    }

    static void BM_Transpose_Int32(benchmark::State& state) {
        const int size = static_cast<int>(state.range(0));

        std::mt19937_64 rng(42);
        Tensor input =
            Tensor::from_random(make_shape(size, size), rng, TensorOptions().dtype(Dtype::Int32))
                .unwrap();
        Tensor output;

        for (auto _ : state) {
            input.transpose(output);
            benchmark::DoNotOptimize(output);
            benchmark::ClobberMemory();
        }

        const int64_t elements = static_cast<int64_t>(size) * size;
        state.SetItemsProcessed(state.iterations() * elements);
        state.SetBytesProcessed(state.iterations() * elements * sizeof(int32_t));
    }

    // Small matrices (should not use cache blocking)
    BENCHMARK(BM_Transpose_Small)->Arg(4)->Arg(8)->Arg(16)->Arg(32)->Unit(benchmark::kMicrosecond);

    // Medium matrices (cache blocking starts to matter)
    BENCHMARK(BM_Transpose_Medium)
        ->Arg(64)
        ->Arg(128)
        ->Arg(256)
        ->Arg(512)
        ->Unit(benchmark::kMicrosecond);

    // Large matrices (cache blocking critical)
    BENCHMARK(BM_Transpose_Large)->Arg(1024)->Arg(2048)->Arg(4096)->Unit(benchmark::kMillisecond);

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

}  // namespace
}  // namespace p10

BENCHMARK_MAIN();
