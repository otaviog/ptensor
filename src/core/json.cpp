#include "json.hpp"

#include <cstddef>
#include <span>
#include <string>

#include "base64.hpp"
#include "compress_staging.hpp"
#include "tensor.hpp"

namespace {
const char* get_compression_string(bool has_compression);
}  // namespace

namespace p10 {
namespace {
    // Owns the text handed to the debugger until the next call on this thread.
    thread_local std::string g_json_debug_buffer;
    thread_local JsonStaging g_json_staging;
}  // namespace

const char* to_json_debug(const Tensor& tensor) {
#ifdef __cpp_exceptions
    try {
        g_json_debug_buffer = std::format("{}", g_json_staging.get_encoder(tensor));
    } catch (const std::exception& error) {
        // A debugger call must not unwind into the debugger.
        g_json_debug_buffer = std::format(R"({{"error":"{}"}})", error.what());
    }
#else
    g_json_debug_buffer = std::format("{}", Json(tensor));
#endif

    return g_json_debug_buffer.c_str();
}

JsonStaging::JsonStaging() noexcept = default;

JsonStaging::JsonStaging(JsonStaging&&) noexcept = default;

JsonStaging& JsonStaging::operator=(JsonStaging&&) noexcept = default;

JsonStaging::~JsonStaging() = default;

detail::CompressStaging* JsonStaging::get_compress() {
    if (!compress_) {
        compress_ = std::make_unique<detail::CompressStaging>();
    }
    return compress_.get();
}

}  // namespace p10

std::format_context::iterator
std::formatter<p10::Json>::format(const p10::Json& target, std::format_context& ctx) {
    const auto& tensor = target.tensor_;
    const std::span<const std::byte> raw = tensor.as_bytes();
    const std::span<const std::byte> blob =
        target.compress_ != nullptr ? target.compress_->compress(raw) : raw;

    return std::format_to(
        ctx.out(),
        R"({{"dtype":"{}","shape":{},"stride":{},"size_bytes":{},"encoding":"{}","blob":"{}"}})",
        p10::to_string(tensor.dtype()),
        p10::to_string(tensor.shape()),
        p10::to_string(tensor.stride()),
        raw.size(),
        get_compression_string(target.compress_ != nullptr),
        p10::Base64(blob)
    );
}

namespace {
const char* get_compression_string(bool has_compression) {
    return has_compression ? "base64+zstd" : "base64";
}
}  // namespace
