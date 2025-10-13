#include <ptensor/op/blur.hpp>
#include <ptensor/tensor.hpp>
#include <ptensor/tensor_wrapper.hpp>

namespace {
int apply_blur(const p10::Tensor& input, p10::Tensor& output);
}

extern "C" int apply_blur(Ptensor input, Ptensor output) {
    return ::apply_blur(p10::unwrap(input), p10::unwrap(output));
}

namespace {
int apply_blur(const p10::Tensor& input, p10::Tensor& output) {
    auto blur = p10::op::GaussianBlur::create(13, 1.5f).unwrap();
    ;
    blur(input, output);
    return 0;
}
}  // namespace
