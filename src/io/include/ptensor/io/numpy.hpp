#include <map>
#include <string>

#include <ptensor/tensor.hpp>

namespace p10::io {
using TensorMap = std::map<std::string, Tensor, std::less<>>;
P10Error save_npz(const std::string& filename, const TensorMap& tensors);
P10Result<TensorMap> load_npz(const std::string& filename);
}  // namespace p10::io
