#pragma once
#include <span>
#include <string>

namespace p10 {
std::string to_base64(std::span<const std::byte> bytes);
}
