#include "json.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <span>
#include <string>
#include <string_view>

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
        g_json_debug_buffer.clear();
        append_json(g_json_staging.get_encoder(tensor), g_json_debug_buffer);
    } catch (const std::exception& error) {
        // A debugger call must not unwind into the debugger.
        g_json_debug_buffer = std::format(R"({{"error":"{}"}})", error.what());
    }
#else
    g_json_debug_buffer.clear();
    append_json(g_json_staging.get_encoder(tensor), g_json_debug_buffer);
#endif

    return g_json_debug_buffer.c_str();
}

void append_json(const Json& encoder, std::string& out) {
    const Tensor& tensor = encoder.tensor_;
    const std::span<const std::byte> raw = tensor.as_bytes();
    const std::span<const std::byte> blob =
        encoder.compress_ != nullptr ? encoder.compress_->compress(raw) : raw;

    // Only the header goes through std::format; the blob, which is the bulk of
    // the object, is appended straight into `out`.
    std::format_to(
        std::back_inserter(out),
        R"({{"dtype":"{}","shape":{},"stride":{},"size_bytes":{},"encoding":"{}","blob":")",
        p10::to_string(tensor.dtype()),
        p10::to_string(tensor.shape()),
        p10::to_string(tensor.stride()),
        raw.size(),
        get_compression_string(encoder.compress_ != nullptr)
    );
    base64_append(blob, out);
    out += R"("})";
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
    std::string encoded;
    p10::append_json(target, encoded);
    return std::ranges::copy(std::string_view(encoded), ctx.out()).out;
}

namespace {
const char* get_compression_string(bool has_compression) {
    return has_compression ? "base64+zstd" : "base64";
}
}  // namespace
