#pragma once

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <ptensor/p10_error.hpp>
#include <ptensor/p10_result.hpp>

namespace p10::testing {

struct ErrorMatcher: Catch::Matchers::MatcherBase<P10Error> {
    explicit ErrorMatcher(P10Error expected) : expected_(std::move(expected)) {}

    template<typename T>
    bool match(const P10Result<T>& actual) const {
        if (actual.is_ok() && expected_.is_ok()) {
            return true;
        }
        return match(actual.error());
    }

    template<typename T>
    bool match(const OkTypeDeduct<T>&) const {
        return expected_.is_ok();
    }

    bool match(const P10Error& actual) const override {
        actual_ = actual;
        return actual.code() == expected_.code();
    }

  protected:
    std::string describe() const override {
        return std::format(
            "Matches P10Error with code {} but got error {}",
            std::to_string(expected_.code()),
            actual_.to_string()
        );
    }

  private:
    mutable P10Error actual_;
    P10Error expected_;
};

inline ErrorMatcher is_error(const P10Error& expected) {
    return ErrorMatcher {expected};
}

struct IsOkMatcher: Catch::Matchers::MatcherBase<P10Error> {
    IsOkMatcher() = default;

    template<typename T>
    bool match(const P10Result<T>& actual) const {
        if (actual.is_ok()) {
            return true;
        }
        return match(actual.error());
    }

    template<typename T>
    bool match(const OkTypeDeduct<T>&) const {
        return true;
    }

    bool match(const P10Error& actual) const override {
        actual_ = actual;
        return actual.is_ok();
    }

  protected:
    std::string describe() const override {
        return "Got error. " + actual_.to_string();
    }

  private:
    mutable P10Error actual_;
};

inline IsOkMatcher is_ok() {
    return {};
}

}  // namespace p10::testing
