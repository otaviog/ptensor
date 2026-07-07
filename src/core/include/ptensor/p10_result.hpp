#pragma once

#include <utility>
#include <variant>

#include <type_traits>

#include "p10_error.hpp"

namespace p10 {

template<class OkType>
class P10Result;

template<typename OkType>
struct OkTypeDeduct;

template<typename OkType>
OkTypeDeduct<std::remove_cvref_t<OkType>> Ok(OkType&& value);

template<class OkType>
class P10Result {
  public:
    P10Result() = default;

    explicit P10Result(P10Error&& error) : value_(std::move(error)) {}

    bool is_ok() const {
        return std::holds_alternative<OkType>(value_);
    }

    bool is_error() const {
        return std::holds_alternative<P10Error>(value_);
    }

    bool has_value() const {
        return value_.index() != std::variant_npos;
    }

    OkType unwrap() {
        if (is_ok()) {
            return std::move(std::get<OkType>(value_));
        }

        error().expect("unwrap called on an error value");
        // This line should never be reached because err().expect() will panic
        return std::move(std::get<OkType>(value_));
    }

    OkType expect(const std::string& message) {
        if (is_ok()) {
            return std::move(std::get<OkType>(value_));
        }
        error().expect(message);
        // This line should never be reached because err().expect() will panic
        return std::move(std::get<OkType>(value_));
    }

    P10Error error() const {
        if (is_error()) {
            return std::get<P10Error>(value_);
        }
        return P10Error::Ok;
    }

    P10Error unwrap_err() {
        return std::move(std::get<P10Error>(value_));
    }

    template<typename Func>
    auto map(Func&& func) -> P10Result<decltype(func(std::declval<OkType>()))> {
        if (is_ok()) {
            return Ok(std::forward<Func>(func)(unwrap()));
        }
        return Err(error());
    }

  private:
    explicit P10Result(const OkType& value) : value_(value) {}

    explicit P10Result(OkType&& value) : value_(std::move(value)) {}

    explicit P10Result(const P10Error& error) : value_(error) {}

  private:
    std::variant<OkType, P10Error> value_;

    template<typename T>
    friend struct OkTypeDeduct;

    template<typename T>
    friend P10Result<T> Err(P10Error&& error);

    template<typename T>
    friend P10Result<T> Err(P10Error::Code error);

    template<typename T>
    friend P10Result<T> Err(P10Error::Code err_code, const std::u8string_view& message);
};

/// Helper class that defers the P10Result<T> type until the conversion site, so
/// `Ok(derived_ptr)` can be returned from a function declared to yield
/// `P10Result<base_ptr>` without an explicit template argument.
template<typename OkType>
struct OkTypeDeduct {
    OkType value;

    template<typename U>
    operator P10Result<U>() && {
        return P10Result<U> {static_cast<U>(std::move(value))};
    }
};

template<typename OkType>
OkTypeDeduct<std::remove_cvref_t<OkType>> Ok(OkType&& value) {
    return {std::forward<OkType>(value)};
}

/// Helper class for template argument deduction for Err function
/// Unlike Ok, Err needs to deduce the template type from the context
/// so we use this intermediate struct to hold the error and convert it to PtensorResult<T>
struct ErrTypeDeduct {
    P10Error error;

    ErrTypeDeduct(P10Error&& err) : error(std::move(err)) {}

    ErrTypeDeduct(P10Error::Code err_code) : error(err_code) {}

    ErrTypeDeduct(P10Error::Code err_code, const std::string_view& message) :
        error(err_code, message) {}

    template<typename OkType>
    operator P10Result<OkType>() {
        return P10Result<OkType> {std::move(error)};
    }
};

inline ErrTypeDeduct Err(P10Error&& error) {
    return ErrTypeDeduct {std::move(error)};
}

inline ErrTypeDeduct Err(P10Error::Code err_code) {
    return ErrTypeDeduct {err_code};
}

inline ErrTypeDeduct Err(P10Error::Code err_code, const std::string_view& message) {
    return ErrTypeDeduct {err_code, message};
}

template<typename TransferOkType>
inline ErrTypeDeduct Err(const P10Result<TransferOkType>& result) {
    return Err(result.error());
}

}  // namespace p10
