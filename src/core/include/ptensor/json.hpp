#pragma once

#include <format>
#include <memory>
#include <string>

namespace p10 {
namespace detail {
    class CompressStaging;
}

enum class JsonEncodeMode { Base64, Base64Compressed };

class Tensor;
class JsonStaging;

/// Formats a tensor as a JSON object: dtype, shape, stride and the tensor
/// buffer encoded as base64, optionally zstd-compressed before encoding.
///
/// ```cpp
/// const std::string text = std::format("{}", p10::Json(tensor));
/// ```
class Json {
  private:
    friend void append_json(const Json& encoder, std::string& out);

    Json(const Tensor& tensor, detail::CompressStaging* compress_ = nullptr) noexcept :
        tensor_(tensor),
        compress_(compress_) {}

    friend struct std::formatter<Json>;
    friend class JsonStaging;
    const Tensor& tensor_;
    detail::CompressStaging* compress_;
};

/// Appends the JSON object for `encoder`'s tensor to `out`.
///
/// The formatting equivalent (`std::format("{}", encoder)`) pushes the base64
/// blob one character at a time through the format output iterator; this writes
/// it into `out` directly, which is what the tlog send path wants.
void append_json(const Json& encoder, std::string& out);

class JsonStaging {
  public:
    JsonStaging() noexcept;
    JsonStaging(JsonStaging&&) noexcept;
    JsonStaging& operator=(JsonStaging&&) noexcept;
    ~JsonStaging();

    Json get_encoder(const Tensor& tensor, JsonEncodeMode mode = JsonEncodeMode::Base64Compressed) {
        if (mode == JsonEncodeMode::Base64) {
            return Json(tensor);
        }

        return Json(tensor, get_compress());
    }

  private:
    detail::CompressStaging* get_compress();
    std::unique_ptr<detail::CompressStaging> compress_;
};

/// Formats `tensor` as a compressed JSON object into a thread local buffer.
///
/// Meant to be called from a debugger, where a function returning a `const
/// char*` is easier to read than a `std::string`. The pointer stays valid
/// until the next call on the same thread.
const char* to_json_debug(const Tensor& tensor);
}  // namespace p10

template<>
struct std::formatter<p10::Json> {
    static constexpr auto parse(std::format_parse_context& ctx) {
        const auto* iter = ctx.begin();
        if (iter != ctx.end() && *iter != '}') {
            throw std::format_error("p10::Json does not accept a format specifier");
        }
        return iter;
    }

    static std::format_context::iterator format(const p10::Json& target, std::format_context& ctx);
};
