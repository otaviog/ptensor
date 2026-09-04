#include "base64.hpp"

#include <cstddef>
#include <cstdint>


namespace {
    constexpr char BASE64_ALPHABET[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
}  // namespace

std::format_context::iterator
std::formatter<p10::Base64>::format(const p10::Base64 &target, std::format_context &ctx) {
    auto out = ctx.out();
    auto bytes = target.data_;

    const size_t input_size = bytes.size();

    const auto* in = reinterpret_cast<const uint8_t*>(bytes.data());
    size_t o = 0;
    size_t i = 0;

    while (i + 3 <= input_size) {
        const uint32_t triple = (static_cast<uint32_t>(in[i]) << 16)
            | (static_cast<uint32_t>(in[i + 1]) << 8) | static_cast<uint32_t>(in[i + 2]);
        *out++ = BASE64_ALPHABET[(triple >> 18) & 0x3F];
        *out++ = BASE64_ALPHABET[(triple >> 12) & 0x3F];
        *out++ = BASE64_ALPHABET[(triple >> 6) & 0x3F];
        *out++ = BASE64_ALPHABET[triple & 0x3F];
        i += 3;
    }

    if (const size_t remaining = input_size - i; remaining == 1) {
        const uint32_t triple = static_cast<uint32_t>(in[i]) << 16;
        *out++ = BASE64_ALPHABET[(triple >> 18) & 0x3F];
        *out++ = BASE64_ALPHABET[(triple >> 12) & 0x3F];
        *out++ = '=';
        *out++ = '=';
    } else if (remaining == 2) {
        const uint32_t triple =
            (static_cast<uint32_t>(in[i]) << 16) | (static_cast<uint32_t>(in[i + 1]) << 8);
        *out++ = BASE64_ALPHABET[(triple >> 18) & 0x3F];
        *out++ = BASE64_ALPHABET[(triple >> 12) & 0x3F];
        *out++ = BASE64_ALPHABET[(triple >> 6) & 0x3F];
        *out++ = '=';
    }

    return out;
}

