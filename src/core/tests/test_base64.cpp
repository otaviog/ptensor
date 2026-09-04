#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <string_view>

#include "../base64.hpp"
#include "base64_decode.hpp"

namespace p10 {

namespace {
    std::string encode(std::span<const std::byte> bytes);
    std::string encode(std::string_view text);
}  // namespace

TEST_CASE("core::Base64 encodes the RFC 4648 test vectors", "[base64]") {
    auto [input, expected] = GENERATE(
        table<std::string_view, std::string_view>(
            {{"", ""},
             {"f", "Zg=="},
             {"fo", "Zm8="},
             {"foo", "Zm9v"},
             {"foob", "Zm9vYg=="},
             {"fooba", "Zm9vYmE="},
             {"foobar", "Zm9vYmFy"}}
        )
    );

    DYNAMIC_SECTION("Encoding \"" << input << "\"") {
        REQUIRE(encode(input) == expected);
    }
}

TEST_CASE("core::Base64 pads according to the input length", "[base64]") {
    const auto length = GENERATE(1, 2, 3, 4, 5, 6, 7, 8);

    DYNAMIC_SECTION("Input of " << length << " bytes") {
        std::vector<std::byte> bytes(static_cast<size_t>(length), std::byte {0xAB});
        const std::string encoded = encode(bytes);

        // 4 output symbols per 3 input bytes, rounded up.
        REQUIRE(encoded.size() == static_cast<size_t>((length + 2) / 3) * 4);

        const auto expected_padding = (length % 3 == 0) ? 0 : 3 - static_cast<size_t>(length % 3);
        REQUIRE(
            static_cast<size_t>(std::count(encoded.begin(), encoded.end(), '=')) == expected_padding
        );
    }
}

TEST_CASE("core::Base64 round-trips every byte value", "[base64]") {
    std::vector<std::byte> bytes(256);
    for (size_t i = 0; i < bytes.size(); i++) {
        bytes[i] = static_cast<std::byte>(i);
    }

    const std::string encoded = encode(bytes);
    const auto decoded = tests::base64_decode(encoded);

    REQUIRE(decoded.has_value());
    REQUIRE(*decoded == bytes);
}

TEST_CASE("core::Base64 emits only alphabet symbols", "[base64]") {
    constexpr std::string_view ALPHABET =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=";

    std::vector<std::byte> bytes(3 * 64);
    for (size_t i = 0; i < bytes.size(); i++) {
        bytes[i] = static_cast<std::byte>((i * 37 + 11) & 0xFF);
    }

    const std::string encoded = encode(bytes);
    REQUIRE(encoded.find_first_not_of(ALPHABET) == std::string::npos);
}

TEST_CASE("core::Base64 encodes into a wider format string", "[base64]") {
    const std::string text = "foobar";
    const auto bytes = std::span(reinterpret_cast<const std::byte*>(text.data()), text.size());

    REQUIRE(std::format("<{}>", Base64(bytes)) == "<Zm9vYmFy>");
}

namespace {
    std::string encode(std::span<const std::byte> bytes) {
        return std::format("{}", Base64(bytes));
    }

    std::string encode(std::string_view text) {
        return encode(std::span(reinterpret_cast<const std::byte*>(text.data()), text.size()));
    }
}  // namespace

}  // namespace p10
