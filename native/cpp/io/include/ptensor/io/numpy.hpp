#include <map>
#include <string>

#include <ptensor/tensor.hpp>

namespace p10::io {
P10Error save_npz(const std::string& filename, const std::map<std::string, Tensor>& tensors);
P10Result<std::map<std::string, Tensor>> load_npz(const std::string& filename);
}  // namespace p10::io
