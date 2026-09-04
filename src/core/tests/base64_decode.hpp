#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <string_view>

namespace p10::tests {

/// Reference base64 decoder used to check what `p10::Base64` produced.
/// Returns nullopt when the text is not valid base64.
inline std::optional<std::vector<std::byte>> base64_decode(std::string_view text) {
    constexpr std::string_view ALPHABET =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    if (text.size() % 4 != 0) {
        return std::nullopt;
    }

    std::vector<std::byte> result;
    result.reserve(text.size() / 4 * 3);

    for (size_t i = 0; i < text.size(); i += 4) {
        uint32_t quad = 0;
        size_t padding = 0;

        for (size_t j = 0; j < 4; j++) {
            const char symbol = text[i + j];
            if (symbol == '=') {
                // Padding is only valid on the last two symbols of the last quad.
                if (i + 4 != text.size() || j < 2) {
                    return std::nullopt;
                }
                padding++;
                quad <<= 6;
                continue;
            }
            if (padding > 0) {
                return std::nullopt;
            }

            const size_t index = ALPHABET.find(symbol);
            if (index == std::string_view::npos) {
                return std::nullopt;
            }
            quad = (quad << 6) | static_cast<uint32_t>(index);
        }

        result.push_back(static_cast<std::byte>((quad >> 16) & 0xFF));
        if (padding < 2) {
            result.push_back(static_cast<std::byte>((quad >> 8) & 0xFF));
        }
        if (padding < 1) {
            result.push_back(static_cast<std::byte>(quad & 0xFF));
        }
    }

    return result;
}

}  // namespace p10::tests
