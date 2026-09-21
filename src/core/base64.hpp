#pragma once
#include <cstddef>
#include <format>
#include <span>
#include <string>

namespace p10 {

/// Appends the base64 of `data` to `out`, growing it once.
///
/// Preferred over formatting a `Base64`: the formatter has to push one
/// character at a time through the format output iterator, which costs about
/// three times more than writing into a sized buffer.
void base64_append(std::span<const std::byte> data, std::string& out);

class Base64 {
  public:
    Base64(std::span<const std::byte> data) noexcept : data_(data) {}

    friend struct std::formatter<Base64>;

  private:
    std::span<const std::byte> data_;
};

}  // namespace p10

template<>
struct std::formatter<p10::Base64> {
    static constexpr auto parse(std::format_parse_context& ctx) {
        const auto *iter = ctx.begin();
        if (iter != ctx.end() && *iter != '}') {
            throw std::format_error("p10::Base64 does not accept a format specifier");
        }
        return iter;
    }

    static std::format_context::iterator format(const p10::Base64& target, std::format_context& ctx);
};
