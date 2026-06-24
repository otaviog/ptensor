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
                testing::is_ok()
            );
            REQUIRE(signal.shape() == make_shape(1, NUM_FFTS, 2));

            Tensor recovered_frequency;
            REQUIRE_THAT(
                Fft(FftOptions().direction(Fft::Forward)).transform(signal, recovered_frequency),
                testing::is_ok()
            );
            REQUIRE_THAT(
                testing::compare_tensors(recovered_frequency, frequency),
                testing::is_ok()
            );
        }

        SECTION("Should inverse into real and forward FFT correctly") {
            Tensor signal;
            REQUIRE_THAT(
                Fft(FftOptions().direction(Fft::InverseReal)).transform(frequency, signal),
                testing::is_ok()
            );
            REQUIRE(signal.shape() == make_shape(1, (NUM_FFTS - 1) * 2));

            Tensor recovered_frequency;
            REQUIRE_THAT(
                Fft(FftOptions().direction(Fft::ForwardReal).normalize(Fft::ByN))
                    .transform(signal, recovered_frequency),
                testing::is_ok()
            );

            REQUIRE_THAT(
                testing::compare_tensors(recovered_frequency, frequency),
                testing::is_ok()
            );
        }
    }

    SECTION("Should handle batched and large signals") {
        const size_t num_signals = 4;
        const size_t signal_size = 4096;
        const Hz sample_rate = 48000.0;
        auto large_signals = Tensor::zeros(make_shape(num_signals, signal_size)).unwrap();
        SineWaveParams params;
        params.sample_rate(sample_rate).period(0.01);
        const std::array<SineWaveParams, num_signals> waves = {
            params.frequency(16000.0).amplitude(1.0).phase_radians(std::numbers::pi / 2),
            params.frequency(16500.0).amplitude(0.75).phase_radians(std::numbers::pi / 4),
            params.frequency(17000.0).amplitude(0.50).phase_radians(std::numbers::pi / 6),
            params.frequency(17500.0).amplitude(0.25).phase_radians(std::numbers::pi / 8),
        };

        for (size_t signal_idx = 0; signal_idx < 4; ++signal_idx) {
            auto signal = large_signals.select_dimension(0, signal_idx).unwrap();
            REQUIRE_THAT(
                generate_sine_wave(signal_size, Dtype::Float32, waves[signal_idx], signal),
                testing::is_ok()
            );
        }

        SECTION("Forward and inverse real FFT on large signals") {
            Tensor frequency_domain;
            REQUIRE_THAT(
                Fft(FftOptions().direction(Fft::ForwardReal))
                    .transform(large_signals, frequency_domain),
                testing::is_ok()
            );

            Tensor rec_signal;
            REQUIRE_THAT(
                Fft(FftOptions().direction(Fft::InverseReal).normalize(Fft::ByN))
                    .transform(frequency_domain, rec_signal),
                testing::is_ok()
            );
            REQUIRE_THAT(testing::compare_tensors(large_signals, rec_signal), testing::is_ok());
        }
    }

    SECTION("ForwardReal accepts 1D input") {
        constexpr size_t SIGNAL_SIZE = 256;
        constexpr size_t NUM_FFTS = SIGNAL_SIZE / 2 + 1;

        SineWaveParams params;
        params.sample_rate(48000.0).period(0.01).frequency(16000.0).amplitude(1.0);

        Tensor signal_1d;
        REQUIRE_THAT(
            generate_sine_wave(SIGNAL_SIZE, Dtype::Float32, params, signal_1d),
            testing::is_ok()
        );
        REQUIRE(signal_1d.shape() == make_shape(SIGNAL_SIZE));

        SECTION("Produces a [F x 2] frequency tensor") {
            Tensor freq_1d;
            REQUIRE_THAT(
                Fft(FftOptions().direction(Fft::ForwardReal)).transform(signal_1d, freq_1d),
                testing::is_ok()
            );
            REQUIRE(freq_1d.shape() == make_shape(NUM_FFTS, 2));
        }

        SECTION("Matches the [1 x T] batched result") {
            Tensor freq_1d;
            REQUIRE_THAT(
                Fft(FftOptions().direction(Fft::ForwardReal)).transform(signal_1d, freq_1d),
                testing::is_ok()
            );

            auto signal_2d = signal_1d.as_reshape(make_shape(1, SIGNAL_SIZE)).unwrap();
            Tensor freq_2d;
            REQUIRE_THAT(
                Fft(FftOptions().direction(Fft::ForwardReal)).transform(signal_2d, freq_2d),
                testing::is_ok()
            );
            REQUIRE(freq_2d.shape() == make_shape(1, NUM_FFTS, 2));

            auto freq_2d_row = freq_2d.select_dimension(0, 0).unwrap();
            REQUIRE_THAT(testing::compare_tensors(freq_1d, freq_2d_row), testing::is_ok());
        }
    }

    SECTION("ForwardReal rejects more than 2 dimensions") {
        auto signal_3d = Tensor::zeros(make_shape(2, 2, 4)).unwrap();
        Tensor freq;
        REQUIRE_THAT(
            Fft(FftOptions().direction(Fft::ForwardReal)).transform(signal_3d, freq),
            testing::is_error(P10Error::InvalidArgument)
        );
    }
}
}  // namespace p10::op
