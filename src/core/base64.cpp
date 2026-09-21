#include "base64.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace {
    constexpr char BASE64_ALPHABET[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
}  // namespace

namespace p10 {

void base64_append(std::span<const std::byte> data, std::string& out) {
    const size_t input_size = data.size();
    const size_t encoded_size = ((input_size + 2) / 3) * 4;

    // One resize, then a raw write: appending through the string's own
    // push_back would re-check the capacity on every character.
    const size_t start = out.size();
    out.resize(start + encoded_size);
    char* dst = out.data() + start;

    const auto* in = reinterpret_cast<const uint8_t*>(data.data());
    size_t i = 0;
    for (; i + 3 <= input_size; i += 3) {
        const uint32_t triple = (static_cast<uint32_t>(in[i]) << 16)
            | (static_cast<uint32_t>(in[i + 1]) << 8) | static_cast<uint32_t>(in[i + 2]);
        *dst++ = BASE64_ALPHABET[(triple >> 18) & 0x3F];
        *dst++ = BASE64_ALPHABET[(triple >> 12) & 0x3F];
        *dst++ = BASE64_ALPHABET[(triple >> 6) & 0x3F];
        *dst++ = BASE64_ALPHABET[triple & 0x3F];
    }

    if (const size_t remaining = input_size - i; remaining == 1) {
        const uint32_t triple = static_cast<uint32_t>(in[i]) << 16;
        *dst++ = BASE64_ALPHABET[(triple >> 18) & 0x3F];
        *dst++ = BASE64_ALPHABET[(triple >> 12) & 0x3F];
        *dst++ = '=';
        *dst++ = '=';
    } else if (remaining == 2) {
        const uint32_t triple =
            (static_cast<uint32_t>(in[i]) << 16) | (static_cast<uint32_t>(in[i + 1]) << 8);
        *dst++ = BASE64_ALPHABET[(triple >> 18) & 0x3F];
        *dst++ = BASE64_ALPHABET[(triple >> 12) & 0x3F];
        *dst++ = BASE64_ALPHABET[(triple >> 6) & 0x3F];
        *dst++ = '=';
    }
}

}  // namespace p10

std::format_context::iterator
std::formatter<p10::Base64>::format(const p10::Base64& target, std::format_context& ctx) {
    // Encodes into a buffer and copies it out: the format iterator takes one
    // character per call, so encoding straight into it is the slow path.
    std::string encoded;
    p10::base64_append(target.data_, encoded);
    return std::ranges::copy(std::string_view(encoded), ctx.out()).out;
}
