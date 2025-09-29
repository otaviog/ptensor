#pragma once

#include <string>

#include <ptensor/ptensor_error.h>

namespace p10 {

class PtensorError {
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

    static PtensorError fromAssert(std::string_view message, std::string_view file, int line);

    PtensorError() = default;

    PtensorError(Code code) : code_(code) {}

    PtensorError(Code code, std::string_view message) : code_(code), message_(message) {}

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
  private:
    Code code_ = Ok;
    std::string message_;
};

inline PtensorError operator<<(PtensorError::Code code, std::string_view message) {
    return PtensorError(code, message);
}
}  // namespace ptensor
