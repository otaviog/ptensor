#include "ptensor/op/elemwise.hpp"

#include <algorithm>

#include "ptensor/tensor.hpp"

namespace p10::op {
namespace {
    template<typename T>
    void add_elemwise_impl(const T* a, const T* b, T* out, size_t size);
    template<typename T>
    void subtract_elemwise_impl(const T* a, const T* b, T* out, size_t size);
}  // namespace

P10Error add_elemwise(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.shape() != b.shape()) {
        return P10Error::InvalidArgument << "Input tensors must have the same shape";
    }
    if (a.dtype() != b.dtype()) {
        return P10Error::InvalidArgument << "Input tensors must have the same data type";
    }
    if (auto status = out.create(a.shape(), a.dtype()); !status.is_ok()) {
        return status;
    }
    a.visit([&](auto a_span) {
        using SpanType = decltype(a_span)::value_type;
        add_elemwise_impl(
            a_span.data(),
            b.as_span1d<SpanType>().unwrap().data(),
            out.as_span1d<SpanType>().unwrap().data(),
            a.size()
        );
    });
    return P10Error::Ok;
}

P10Error subtract_elemwise(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.shape() != b.shape()) {
        return P10Error::InvalidArgument << "Input tensors must have the same shape";
    }
    if (a.dtype() != b.dtype()) {
        return P10Error::InvalidArgument << "Input tensors must have the same data type";
    }
    out.create(a.shape(), a.dtype());

    a.visit([&](auto a_span) {
        using SpanType = decltype(a_span)::value_type;

        subtract_elemwise_impl(
            a_span.data(),
            b.as_span1d<SpanType>().unwrap().data(),
            out.as_span1d<SpanType>().unwrap().data(),
            a.size()
        );
    });
    return P10Error::Ok;
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