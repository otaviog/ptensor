#include "compress_staging.hpp"

#include <zstd.h>

#include "detail/panic.hpp"

namespace p10::detail {
std::span<const std::byte> CompressStaging::compress(std::span<const std::byte> data) {
    const size_t bound = ZSTD_compressBound(data.size_bytes());
    if (buffer_.size() < bound) {
        buffer_.resize(bound);
    }

    const size_t compressed_size =
        ZSTD_compress(buffer_.data(), buffer_.size(), data.data(), data.size_bytes(), 1);
    if (ZSTD_isError(compressed_size) != 0) {
        detail::panic(ZSTD_getErrorName(compressed_size));
    }

    return std::span<const std::byte>(buffer_).first(compressed_size);
}
}  // namespace p10::detail
