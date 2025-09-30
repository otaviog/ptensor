#include <string>

#include <ptensor/tensor.hpp>

namespace p10::io {
PtensorResult<Tensor> load_image(const std::string& path);
PtensorError save_image(const std::string& path, const Tensor& tensor);
}  // namespace p10::io
