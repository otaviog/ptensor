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
            return std::abs(a - b) < static_cast<float>(options.tolerance());
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
    const auto match_count = t1.dtype().match([&](auto type_tag) -> size_t {
        using scalar_t = decltype(type_tag)::type;
        auto it1 = t1.iterator<scalar_t>().unwrap();
        auto it2 = t2.iterator<scalar_t>().unwrap();
        size_t match_count = 0;
        
        while(it1.has_next()) {
            
            const auto [val1, _1] = it1.next();
            const auto [val2, _2] = it2.next();
            if (compare.equal(val1, val2)) {
                ++match_count;
            }
            
        }
        return match_count;
    });
    
    if (match_count != t1.size()) {
        return P10Error::AssertionError << "Data are different"
                                        << std::string("Match rate is ")
            + std::to_string(static_cast<double>(match_count) / t1.size() * 100.0) + "%";
    }
    return P10Error::Ok;
}

}  // namespace p10::testing
