#include "elementary.hpp"

#include <algorithm>

#include <ptensor/tensor.hpp>

namespace p10::recog {

P10Error subtract_elements(Tensor& a, double rhs_) {
    if (!a.is_contiguous()) {
        return P10Error::InvalidArgument << "Tensor must be contiguous for subtract_elements";
    }
    
    a.visit([rhs_](auto span) {
        using scalar_t = typename decltype(span)::value_type;
        const auto rhs = static_cast<scalar_t>(rhs_);
        std::transform(span.begin(), span.end(), [rhs](const auto value) { return value - rhs; });
    });

    return P10Error::Ok;
}
}  // namespace p10::recog
