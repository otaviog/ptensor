#pragma once

#include <ptensor/tensor.hpp>

namespace p10::testing {

class CompareOptions {
  public:
    CompareOptions& tolerance(double tol) {
        tolerance_ = tol;
        return *this;
    }

    double tolerance() const {
        return tolerance_;
    }

  private:
    double tolerance_ = 1e-6;
};

namespace detail {
    struct Compare {
        Compare(const CompareOptions& options) : options(options) {}

        bool equal(float a, float b) const {
            return std::abs(a - b) < float(options.tolerance());
        }

        bool equal(double a, double b) const {
            return std::abs(a - b) < options.tolerance();
        }

        template<typename T>
        bool equal(T a, T b) const {
            return a == b;
        }

        CompareOptions options;
    };
}  // namespace detail

inline P10Error
compare_tensors(const Tensor& t1, const Tensor& t2, const CompareOptions& options = {}) {
    if (t1.shape() != t2.shape()) {
        return P10Error::AssertionError
            << (std::string("Shapes are different") + "Shape 1: " + to_string(t1.shape())
                + "\nShape 2: " + to_string(t2.shape()));
    }

    if (t1.stride() != t2.stride()) {
        return P10Error::AssertionError
            << (std::string("Strides are different") + "Stride 1: " + to_string(t1.stride())
                + "\nStride 2: " + to_string(t2.stride()));
    }

    if (t1.dtype() != t2.dtype()) {
        return P10Error::AssertionError
            << (std::string("Data types are different") + "Data type 1: " + to_string(t1.dtype())
                + "\nData type 2: " + to_string(t2.dtype()));
    }

    if (t1.size_bytes() != t2.size_bytes()) {
        return P10Error::AssertionError
            << (std::string("Sizes (byte) are different") + "Size 1: "
                + std::to_string(t1.size_bytes()) + "\nSize 2: " + std::to_string(t2.size_bytes()));
    }

    detail::Compare compare(options);
    const auto match_count = t1.visit([&t2, compare](auto data1) {
        using scalar_t = decltype(data1)::value_type;
        const auto data2 = t2.as_span1d<scalar_t>().unwrap();

        size_t match_count = 0;
        auto it1 = data1.begin();
        auto it2 = data2.begin();

        while (it1 != data1.end()) {
            if (compare.equal(*it1, *it2)) {
                ++match_count;
            }
            ++it1;
            ++it2;
        }
        return match_count;
    });

    if (match_count != t1.size()) {
        return P10Error::AssertionError << "Data are different"
                                        << std::string("Match rate is ")
            + std::to_string(static_cast<double>(match_count) / t1.size());
    }
    return P10Error::Ok;
}

}  // namespace p10::testing
