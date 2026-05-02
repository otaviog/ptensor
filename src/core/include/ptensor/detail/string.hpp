#pragma once

#include <string>

#include "../p10_result.hpp"

namespace p10::detail {
P10Result<std::wstring> string_to_wstring(const std::string& ansi);
}  // namespace p10::detail