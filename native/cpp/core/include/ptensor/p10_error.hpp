#pragma once

#include <string>

#include <ptensor/ptensor_error.h>
#include "detail/panic.hpp"

namespace p10 {

class P10Error {
  public:
    enum Code {
        Ok = P10_OK,  // No error
        UnknownError = P10_UNKNOWN_ERROR,  // Unknown error
        AssertionError = P10_ASSERTION_ERROR,  // Assertion error
        InvalidArgument = P10_INVALID_ARGUMENT,  // Invalid argument
        InvalidOperation = P10_INVALID_OPERATION,  // Invalid operation
        OutOfMemory = P10_OUT_OF_MEMORY,  // Out of memory
        OutOfRange = P10_OUT_OF_RANGE,  // Out of range
        NotImplemented = P10_NOT_IMPLEMENTED,  // Not implemented
        OsError = P10_OS_ERROR,
        IoError = P10_IO_ERROR
    };

    static P10Error fromAssert(std::string_view message, std::string_view file, int line);

#ifdef PTENSOR_HAS_WINDOWS_H
    static P10Error from_win32_error(unsigned long error_code);
#endif

    P10Error() = default;

    P10Error(Code code) : code_(code) {}

    P10Error(Code code, std::string_view message) : code_(code), message_(message) {}

    constexpr operator Code() const {
        return code_;
    }

    explicit operator bool() const = delete;

    Code code() const {
        return static_cast<Code>(code_);
    }

    std::string to_string() const;

    bool is_ok() const {
        return code_ == Ok;
    }

    bool is_error() const {
        return !is_ok();
    }

    void expect(const std::string& message) const {
        if (is_error()) {
            detail::panic((message + ":" + to_string()).data());
        }
    }

  private:
    Code code_ = Ok;
    std::string message_;
};

inline P10Error operator<<(P10Error::Code code, std::string_view message) {
    return P10Error(code, message);
}
}  // namespace p10
