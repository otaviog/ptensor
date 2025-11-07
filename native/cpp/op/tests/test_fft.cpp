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
    std::array<std::complex<float>, 8> frequency_data = {
        std::complex<float> {1.0f, 0.0f},
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

    const size_t signal_size = (frequency_data.size() - 1) * 2;
    Fft ifft(signal_size, FftOptions().direction(Fft::Inverse).normalize(Fft::ByN));
    Tensor signal;

    REQUIRE(ifft.transform(frequency, signal).is_ok());
    REQUIRE(signal.shape() == make_shape(1, signal_size));
    std::cout << "Time-domain signal:\n " << p10::to_string(signal) << std::endl;

    // Now perform forward FFT
    Fft fft(signal_size, Fft::Forward);
    Tensor recovered_frequency;

    REQUIRE(fft.transform(signal, recovered_frequency).is_ok());
    std::cout << "Frequency-domain signal:\n " << p10::to_string(recovered_frequency) << std::endl;
    REQUIRE_THAT(testing::compare_tensors(recovered_frequency, frequency), testing::IsOk());
}
}  // namespace p10::op