#include "ptensor_error.hpp"

#ifdef PTENSOR_HAS_WINDOWS_H
    #include <Windows.h>

    #include <system_error>
#endif

namespace p10 {

PtensorError PtensorError::fromAssert(std::string_view message, std::string_view file, int line) {
    return PtensorError(
        AssertionError,
        std::string(message) + " (" + std::string(file) + ":" + std::to_string(line) + ")"
    );
}

#ifdef PTENSOR_HAS_WINDOWS_H
PtensorError PtensorError::from_win32_error(unsigned long error_code) {
    if (error_code == ERROR_SUCCESS) {
        return PtensorError::Ok;
    }
    std::error_code ec(error_code, std::system_category());
    return PtensorError::OsError << ec.message();
}
#endif

std::string PtensorError::to_string() const {
    std::string str;
    switch (code_) {
        case Ok:
            str = "No error";
            break;
        case UnknownError:
            str = "Unknown error";
            break;
        case AssertionError:
            str = "Assertion error";
            break;
        case InvalidArgument:
            str = "Invalid argument";
            break;
        case InvalidOperation:
            str = "Invalid operation";
            break;
        case OutOfMemory:
            str = "Out of memory";
            break;
        case OutOfRange:
            str = "Out of range";
            break;
        case NotImplemented:
            str = "Not implemented";
            break;
        case OsError:
            str = "OS error";
            break;
        case IoError:
            str = "IO error";
            break;
        default:
            str = "Unknown";
            break;
    }

    if (message_.empty()) {
        return str;
    }

    return str + ": " + message_;
}

}  // namespace p10
