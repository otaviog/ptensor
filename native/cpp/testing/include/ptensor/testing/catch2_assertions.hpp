#pragma once

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <ptensor/p10_error.hpp>
#include <ptensor/p10_result.hpp>

namespace p10::testing {

struct ErrorMatcher: Catch::Matchers::MatcherBase<P10Error> {
    ErrorMatcher(P10Error expected) : expected_(expected) {}

    template<typename T>
    bool match(const P10Result<T>& actual) const {
        if (actual.is_ok() && expected_.is_ok()) {
            return true;
        }
        return match(actual.error());
    }

    bool match(const P10Error& actual) const override {
        actual_ = actual;
        return actual.code() == expected_.code();
    }

    std::string describe() const override {
        return "Matches P10Error with code " + std::to_string(expected_.code()) + " but got error "
            + actual_.to_string();
    }

  private:
    mutable P10Error actual_;
    P10Error expected_;
};

inline ErrorMatcher IsError(const P10Error& expected) {
    return ErrorMatcher(expected);
}

struct IsOkMatcher: Catch::Matchers::MatcherBase<P10Error> {
    IsOkMatcher() {}

    template<typename T>
    bool match(const P10Result<T>& actual) const {
        if (actual.is_ok()) {
            return true;
        }
        return match(actual.error());
    }

    bool match(const P10Error& actual) const override {
        actual_ = actual;
        return actual.is_ok();
    }

    std::string describe() const override {
        return "Got error. " + actual_.to_string();
    }

  private:
    mutable P10Error actual_;
};

inline IsOkMatcher IsOk() {
    return IsOkMatcher();
}

}  // namespace p10::testing
