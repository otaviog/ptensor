#pragma once

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <ptensor/ptensor_error.hpp>
#include <ptensor/ptensor_result.hpp>
#include <ptensor/tensor.hpp>

namespace p10::testing {

template<typename Type>
inline void require_ok(const PtensorResult<Type>& result) {
    if (!result.is_ok()) {
        const auto err1 = result.err().to_string();
        const std::string err(err1.begin(), err1.end());
        FAIL(err);
    }
}

struct IsOkMatcher: Catch::Matchers::MatcherGenericBase {
    template<typename T>
    bool match(const PtensorResult<T>& result) const {
        return result.is_ok();
    }

    bool match(const PtensorError& result) const {
        return result.is_ok();
    }

    std::string describe() const override {
        return "Test if the result is Ok";
    }
};

inline IsOkMatcher IsOk() {
    return IsOkMatcher();
}

struct IsErrMatcher: Catch::Matchers::MatcherGenericBase {
    template<typename T>
    bool match(const PtensorResult<T>& result) const {
        return !result.is_ok();
    }

    bool match(const PtensorError& result) const {
        return !result.is_ok();
    }

    std::string describe() const override {
        return "Test if the result is an error (not OK)";
    }
};

inline IsErrMatcher IsErr() {
    return IsErrMatcher();
}

struct Compare {
    static bool equal(float a, float b) {
        return std::abs(a - b) < 1e-6;
    }

    static bool equal(double a, double b) {
        return std::abs(a - b) < 1e-6;
    }

    template<typename T>
    static bool equal(T a, T b) {
        return a == b;
    }
};

inline PtensorError compare_tensors(const Tensor& t1, const Tensor& t2) {
    if (t1.shape() != t2.shape()) {
        return PtensorError::AssertionError
            << (std::string("Shapes are different") + "Shape 1: " + to_string(t1.shape())
                + "\nShape 2: " + to_string(t2.shape()));
    }

    if (t1.stride() != t2.stride()) {
        return PtensorError::AssertionError
            << (std::string("Strides are different") + "Stride 1: " + to_string(t1.stride())
                + "\nStride 2: " + to_string(t2.stride()));
    }

    if (t1.dtype() != t2.dtype()) {
        return PtensorError::AssertionError
            << (std::string("Data types are different") + "Data type 1: " + to_string(t1.dtype())
                + "\nData type 2: " + to_string(t2.dtype()));
    }

    if (t1.size_bytes() != t2.size_bytes()) {
        return PtensorError::AssertionError
            << (std::string("Sizes (byte) are different") + "Size 1: "
                + std::to_string(t1.size_bytes()) + "\nSize 2: " + std::to_string(t2.size_bytes()));
    }

    const auto match_count = t1.visit([&t2](auto data1) {
        using scalar_t = decltype(data1)::value_type;
        const auto data2 = t2.as_span1d<scalar_t>().unwrap();

        size_t match_count = 0;
        auto it1 = data1.begin();
        auto it2 = data2.begin();
        while (it1 != data1.end()) {
            if (Compare::equal(*it1, *it2)) {
                ++match_count;
            }
            ++it1;
            ++it2;
        }
        return match_count;
    });

    if (match_count != t1.size()) {
        return PtensorError::AssertionError << "Data are different"
                                            << std::string("Match rate is ")
            + std::to_string(static_cast<double>(match_count) / t1.size());
    }
    return PtensorError::Ok;
}

}  // namespace p10::testing
