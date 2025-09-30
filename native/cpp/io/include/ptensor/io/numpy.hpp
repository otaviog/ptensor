#include <map>
#include <string>

#include <ptensor/tensor.hpp>

namespace p10::io {
PtensorError save_npz(const std::string& filename, const std::map<std::string, Tensor>& tensors);
PtensorResult<std::map<std::string, Tensor>> load_npz(const std::string& filename);
}  // namespace p10::io
