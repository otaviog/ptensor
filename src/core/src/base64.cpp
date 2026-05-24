#include "base64.hpp"

#include <cstddef>
#include <cstdint>

namespace p10 {

namespace {
    constexpr char BASE64_ALPHABET[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
}  // namespace

std::string to_base64(std::span<const std::byte> bytes) {
    const size_t input_size = bytes.size();
    const size_t output_size = ((input_size + 2) / 3) * 4;

    std::string out;
    out.resize(output_size);

    const auto* in = reinterpret_cast<const uint8_t*>(bytes.data());
    size_t o = 0;
    size_t i = 0;

    while (i + 3 <= input_size) {
        const uint32_t triple = (uint32_t(in[i]) << 16) | (uint32_t(in[i + 1]) << 8) | uint32_t(in[i + 2]);
        out[o++] = BASE64_ALPHABET[(triple >> 18) & 0x3F];
        out[o++] = BASE64_ALPHABET[(triple >> 12) & 0x3F];
        out[o++] = BASE64_ALPHABET[(triple >> 6) & 0x3F];
        out[o++] = BASE64_ALPHABET[triple & 0x3F];
        i += 3;
    }

    if (const size_t remaining = input_size - i; remaining == 1) {
        const uint32_t triple = uint32_t(in[i]) << 16;
        out[o++] = BASE64_ALPHABET[(triple >> 18) & 0x3F];
        out[o++] = BASE64_ALPHABET[(triple >> 12) & 0x3F];
        out[o++] = '=';
        out[o++] = '=';
    } else if (remaining == 2) {
        const uint32_t triple = (uint32_t(in[i]) << 16) | (uint32_t(in[i + 1]) << 8);
        out[o++] = BASE64_ALPHABET[(triple >> 18) & 0x3F];
        out[o++] = BASE64_ALPHABET[(triple >> 12) & 0x3F];
        out[o++] = BASE64_ALPHABET[(triple >> 6) & 0x3F];
        out[o++] = '=';
    }

    return out;
}

}  // namespace p10
