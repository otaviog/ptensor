#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace p10::detail {

/// Reusable staging buffer for zstd compression.
///
/// The returned span points into the instance's buffer and is invalidated by
/// the next call to `compress()` or by the instance's destruction.
class CompressStaging {
  public:
    std::span<const std::byte> compress(std::span<const std::byte> data);

  private:
    std::vector<std::byte> buffer_;
};

}  // namespace p10::detail
