#include <complex>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <ptensor/io/image.hpp>
#include <ptensor/op/fft.hpp>

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

    FFT fft(frequency_data.size(), FFT::Inverse);
    Tensor signal;

    REQUIRE(fft.transform(frequency, signal).is_ok());
    REQUIRE(signal.shape() == make_shape(1, 14, 2));

    // Now perform forward FFT
    FFT ifft(frequency_data.size(), FFT::Forward);
    Tensor recovered_frequency;
    REQUIRE(ifft.transform(signal, recovered_frequency).is_ok());
    REQUIRE(recovered_frequency.shape() == make_shape(frequency_data.size(), 2));
    auto recovered_span = recovered_frequency.as_span1d<std::complex<float>>().unwrap();
    for (size_t i = 0; i < frequency_data.size(); i++) {
        REQUIRE(std::abs(recovered_span[i] - frequency_data[i]) < 1e-5f);
    }
}
}  // namespace p10::op