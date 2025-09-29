#pragma once

#include <utility>
#include <variant>

#include "detail/panic.hpp"
#include "ptensor_error.hpp"

namespace p10 {
template<class OkType>
class PtensorResult {
  public:
    PtensorResult() = default;

    bool is_ok() const {
        return std::holds_alternative<OkType>(value_);
    }

    bool has_value() const {
        return value_.index() != std::variant_npos;
    }

    OkType unwrap() {
        return std::move(std::get<OkType>(value_));
    }

    OkType expect(const std::string& message) {
        if (is_ok()) {
            return unwrap();
        } else {
            std::string msg(err().to_string());
            detail::panic((message + ":" + msg).data());
        }
    }

    const PtensorError& err() const {
        return std::get<PtensorError>(value_);
    }

    PtensorError unwrap_err() {
        return std::move(std::get<PtensorError>(value_));
    }

    explicit PtensorResult(PtensorError&& error) : value_(std::move(error)) {}

  private:
    explicit PtensorResult(const OkType& value) : value_(value) {}

    explicit PtensorResult(OkType&& value) : value_(std::move(value)) {}

  private:
    std::variant<OkType, PtensorError> value_;

    template<typename T>
    friend PtensorResult<T> Ok(T&& value);

    template<typename T>
    friend PtensorResult<T> Err(PtensorError&& error);

    template<typename T>
    friend PtensorResult<T> Err(PtensorError::Code error);

    template<typename T>
    friend PtensorResult<T> Err(PtensorError::Code err_code, const std::u8string_view& message);
};

template<typename OkType>
PtensorResult<OkType> Ok(OkType&& value) {
    return PtensorResult<OkType> {std::forward<OkType>(value)};
}

/// Helper class for template argument deduction for Err function
/// Unlike Ok, Err needs to deduce the template type from the context
/// so we use this intermediate struct to hold the error and convert it to PtensorResult<T>
struct ErrTypeDeduct {
    PtensorError error;

    ErrTypeDeduct(PtensorError&& err) : error(std::move(err)) {}

    ErrTypeDeduct(PtensorError::Code err_code) : error(err_code) {}

    ErrTypeDeduct(PtensorError::Code err_code, const std::string_view& message) :
        error(err_code, message) {}

    template<typename OkType>
    operator PtensorResult<OkType>() {
        return PtensorResult<OkType> {std::move(error)};
    }
};

inline ErrTypeDeduct Err(PtensorError&& error) {
    return ErrTypeDeduct {std::move(error)};
}

inline ErrTypeDeduct Err(PtensorError::Code err_code) {
    return ErrTypeDeduct {err_code};
}

inline ErrTypeDeduct Err(PtensorError::Code err_code, const std::string_view& message) {
    return ErrTypeDeduct {err_code, message};
}

}  // namespace p10
