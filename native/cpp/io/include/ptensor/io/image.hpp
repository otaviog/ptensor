#include <string>

#include <ptensor/tensor.hpp>

namespace p10::io {
P10Result<Tensor> load_image(const std::string& path);
P10Error save_image(const std::string& path, const Tensor& tensor);
}  // namespace p10::io
