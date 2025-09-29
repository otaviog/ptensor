#include "string.hpp"

#include <Windows.h>

#include "ptensor_error.hpp"

namespace p10 {

PtensorResult<std::wstring> string_to_wstring(const std::string& ansi) {
    ULONG charCount;
    DWORD dwError;
    std::wstring wide;
    charCount = ULONG(ansi.length());
    wide.resize(charCount);
    if (MultiByteToWideChar(CP_ACP, 0, ansi.c_str(), charCount, &wide[0], charCount) == 0) {
        dwError = GetLastError();
        wide.clear();
        return Err(PtensorError::from_win32_error(dwError));
    }
    return Ok(std::move(wide));
}
}  // namespace ptensor
