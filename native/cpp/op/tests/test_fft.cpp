#include <complex>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <ptensor/io/image.hpp>
#include <ptensor/op/fft.hpp>
#include <ptensor/op/wave.hpp>
#include <ptensor/tensor_print.hpp>
#include <ptensor/testing/catch2_assertions.hpp>

namespace p10::op {
TEST_CASE("Op: FFT and IFFT", "[tensorop]") {
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
        Fft ifft(NUM_FFTS, FftOptions().direction(Fft::Inverse).normalize(Fft::ByN));

        Tensor signal;
        REQUIRE(ifft.transform(frequency, signal).is_ok());
        REQUIRE(signal.shape() == make_shape(1, NUM_FFTS, 2));

        Fft fft(NUM_FFTS, Fft::Forward);
        Tensor recovered_frequency;

        REQUIRE(fft.transform(signal, recovered_frequency).is_ok());
        REQUIRE_THAT(testing::compare_tensors(recovered_frequency, frequency), testing::IsOk());
    }

    SECTION("Should inverse into real and forward FFT correctly") {
        Fft ifft(NUM_FFTS, FftOptions().direction(Fft::InverseReal));

        Tensor signal;
        REQUIRE(ifft.transform(frequency, signal).is_ok());
        REQUIRE(signal.shape() == make_shape(1, (NUM_FFTS - 1) * 2));
        Fft fft(NUM_FFTS, Fft::ForwardReal);
        Tensor recovered_frequency;

        REQUIRE(fft.transform(signal, recovered_frequency).is_ok());
        REQUIRE_THAT(testing::compare_tensors(recovered_frequency, frequency), testing::IsOk());
    }

    SECTION("Should handle batched and large signals") {
        SKIP("WIP");
        auto large_signals = Tensor::zeros(make_shape(4, 4048)).unwrap();
        for (size_t i = 0; i < 4; ++i) {
            //generate_sine_wave(4048, Dtype::Float32, SineWaveOptions().frequency(440.0), large_signals.select_dimension(0, i).unwrap());
        }

        Fft fft(4048, FftOptions().direction(Fft::ForwardReal));
        Tensor frequency_domain;
        REQUIRE(fft.transform(large_signals, frequency_domain).is_ok());

        
    }
}
}  // namespace p10::op
