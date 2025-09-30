#include "elemwise.hpp"

#include <algorithm>

#include "tensor.hpp"

namespace p10::op {
namespace {
    template<typename T>
    void add_elemwise_impl(const T* a, const T* b, T* out, size_t size);
    template<typename T>
    void subtract_elemwise_impl(const T* a, const T* b, T* out, size_t size);
}  // namespace

PtensorError add_elemwise(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.shape() != b.shape()) {
        return PtensorError::InvalidArgument << "Input tensors must have the same shape";
    }
    if (a.dtype() != b.dtype()) {
        return PtensorError::InvalidArgument << "Input tensors must have the same data type";
    }
    out.create(a.dtype(), a.shape());
    a.visit_data([&](auto a_span) {
        using SpanType = decltype(a_span)::value_type;
        add_elemwise_impl(a_span.data(), b.data<SpanType>(), out.data<SpanType>(), a.size());
    });
    return PtensorError::OK;
}

PtensorError subtract_elemwise(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.shape() != b.shape()) {
        return PtensorError::InvalidArgument << "Input tensors must have the same shape";
    }
    if (a.dtype() != b.dtype()) {
        return PtensorError::InvalidArgument << "Input tensors must have the same data type";
    }
    out.create(a.dtype(), a.shape());

    a.visit_data([&](auto a_span) {
        using SpanType = decltype(a_span)::value_type;

        subtract_elemwise_impl(a_span.data(), b.data<SpanType>(), out.data<SpanType>(), a.size());
    });
    return PtensorError::OK;
}

namespace {
    template<typename T>
    void add_elemwise_impl(const T* a, const T* b, T* out, size_t size) {
        std::transform(a, a + size, b, out, std::plus<T>());
    }

    template<typename T>
    void subtract_elemwise_impl(const T* a, const T* b, T* out, size_t size) {
        std::transform(a, a + size, b, out, std::minus<T>());
    }
}  // namespace

}  // namespace p10::op