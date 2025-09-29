#pragma once

#include <string>

#include "../ptensor_result.hpp"

namespace p10::detail {
PtensorResult<std::wstring> string_to_wstring(const std::string& ansi);
}  // namespace p10::detail