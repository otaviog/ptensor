#include <complex>
#include <numbers>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <ptensor/io/audio.hpp>
#include <ptensor/io/image.hpp>
#include <ptensor/io/numpy.hpp>
#include <ptensor/op/fft.hpp>
#include <ptensor/op/wave.hpp>
#include <ptensor/op/window_function.hpp>
#include <ptensor/tensor_print.hpp>
#include <ptensor/testing/catch2_assertions.hpp>
#include <ptensor/testing/compare_tensors.hpp>

#include "catch2/catch_approx.hpp"
#include "catch2/matchers/catch_matchers.hpp"

namespace p10::op {
TEST_CASE("Op: FFT and IFFT", "[tensorop]") {
    SECTION("Basic forward and inverse FFT") {
        constexpr size_t NUM_FFTS = 9;
        std::array<std::complex<float>, NUM_FFTS> frequency_data = {
            std::complex<float> {1.0f, 0.0f},
            std::complex<float> {0.0f, 0.0f},
            std::complex<float> {0.0f, 0.0f},
            std::complex<float> {0.0f, 0.0f},
            std::complex<float> {0.0f, 0.0f},
            std::complex<float> {0.0f, 0.0f},
            std::complex<float> {0.0f, 0.0f},
            std::complex<float> {0.0f, 0.0f},
            std::complex<float> {0.0f, 0.0f}
        };
        auto frequency = Tensor::from_data<float>(
            reinterpret_cast<float*>(frequency_data.data()),
            make_shape(1, frequency_data.size(), 2)
        );

        SECTION("Should inverse into complex and forward FFT correctly") {
            SKIP("TODO: not implemented");
            Tensor signal;
            REQUIRE_THAT(
                Fft(FftOptions().direction(Fft::Inverse).normalize(Fft::ByN))
                    .transform(frequency, signal),
                testing::IsOk()
            );
            REQUIRE(signal.shape() == make_shape(1, NUM_FFTS, 2));

            Tensor recovered_frequency;
            REQUIRE_THAT(
                Fft(FftOptions().direction(Fft::Forward)).transform(signal, recovered_frequency),
                testing::IsOk()
            );
            REQUIRE_THAT(testing::compare_tensors(recovered_frequency, frequency), testing::IsOk());
        }

        SECTION("Should inverse into real and forward FFT correctly") {
            Tensor signal;
            REQUIRE_THAT(
                Fft(FftOptions().direction(Fft::InverseReal)).transform(frequency, signal),
                testing::IsOk()
            );
            REQUIRE(signal.shape() == make_shape(1, (NUM_FFTS - 1) * 2));

            Tensor recovered_frequency;
            REQUIRE_THAT(
                Fft(FftOptions().direction(Fft::ForwardReal).normalize(Fft::ByN))
                    .transform(signal, recovered_frequency),
                testing::IsOk()
            );

            REQUIRE_THAT(testing::compare_tensors(recovered_frequency, frequency), testing::IsOk());
        }
    }

    SECTION("Should handle batched and large signals") {
        const size_t NUM_SIGNALS = 4;
        const size_t SIGNAL_SIZE = 4096;
        const Hz SAMPLE_RATE = 48000.0;
        auto large_signals = Tensor::zeros(make_shape(NUM_SIGNALS, SIGNAL_SIZE)).unwrap();
        SineWaveParams params;
        params.sample_rate(SAMPLE_RATE).period(0.01);
        const std::array<SineWaveParams, NUM_SIGNALS> waves = {
            params.frequency(16000.0).amplitude(1.0).phaseRadians(std::numbers::pi / 2),
            params.frequency(16500.0).amplitude(0.75).phaseRadians(std::numbers::pi / 4),
            params.frequency(17000.0).amplitude(0.50).phaseRadians(std::numbers::pi / 6),
            params.frequency(17500.0).amplitude(0.25).phaseRadians(std::numbers::pi / 8),
        };

        for (size_t signal_idx = 0; signal_idx < 4; ++signal_idx) {
            auto signal = large_signals.select_dimension(0, signal_idx).unwrap();
            REQUIRE_THAT(
                generate_sine_wave(SIGNAL_SIZE, Dtype::Float32, waves[signal_idx], signal),
                testing::IsOk()
            );
        }

        SECTION("Forward and inverse real FFT on large signals") {
            Tensor frequency_domain;
            REQUIRE_THAT(
                Fft(FftOptions().direction(Fft::ForwardReal))
                    .transform(large_signals, frequency_domain),
                testing::IsOk()
            );

            Tensor rec_signal;
            REQUIRE_THAT(
                Fft(FftOptions().direction(Fft::InverseReal).normalize(Fft::ByN))
                    .transform(frequency_domain, rec_signal),
                testing::IsOk()
            );
            REQUIRE_THAT(testing::compare_tensors(large_signals, rec_signal), testing::IsOk());
        }
    }
}
}  // namespace p10::op
