#pragma once
#include <string>

#include <ptensor/p10_error.hpp>

namespace p10 {
class Tensor;
}

namespace p10::io {
P10Error load_audio(const std::string& path, Tensor& tensor, int64_t& sample_rate);
P10Error save_audio(const std::string& path, const Tensor& tensor, int64_t sample_rate);
}  // namespace p10::io
