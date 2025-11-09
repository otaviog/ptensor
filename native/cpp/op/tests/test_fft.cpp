#include <complex>
#include <iostream>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <ptensor/io/image.hpp>
#include <ptensor/op/fft.hpp>
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
    //std::cout << "Frequency-domain signal:\n " << p10::to_string(frequency) << std::endl;

    SECTION("Should inverse into complex and forward FFT correctly") {
        Fft ifft(NUM_FFTS, FftOptions().direction(Fft::Inverse).normalize(Fft::ByN));

        Tensor signal;
        REQUIRE(ifft.transform(frequency, signal).is_ok());
        REQUIRE(signal.shape() == make_shape(1, NUM_FFTS, 2));
        // Debug output removed to keep test results clean
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
        // Debug output removed to keep test results clean

        // Now perform forward FFT
        Fft fft(NUM_FFTS, Fft::ForwardReal);
        Tensor recovered_frequency;

        REQUIRE(fft.transform(signal, recovered_frequency).is_ok());
        // Debug output removed to keep test results clean
        REQUIRE_THAT(testing::compare_tensors(recovered_frequency, frequency), testing::IsOk());
    }
}
}  // namespace p10::op
