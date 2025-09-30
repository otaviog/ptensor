#include "numpy.hpp"

#include <algorithm>
#include <cstdint>

#include <cnpy.h>

#include "ptensor/ptensor_error.hpp"
#include "ptensor/ptensor_result.hpp"

namespace p10::io {
using TensorMap = std::map<std::string, Tensor>;

PtensorError save_npz(const std::string& filename, const TensorMap& tensors) {
    std::string mode = "w";
    for (const auto& [key_name, tensor] : tensors) {
        std::vector<size_t> shape;

        std::transform(
            tensor.shape().begin(),
            tensor.shape().end(),
            std::back_inserter(shape),
            [](const int64_t& dim) { return static_cast<size_t>(dim); }
        );
        tensor.visit_data([&](auto span) {
            cnpy::npz_save(filename, key_name, span.data(), shape, mode);
        });
        mode = "a";
    }

    return PtensorError::OK;
}

PtensorResult<TensorMap> load_npz(const std::string& filename) {
    cnpy::npz_t npz = cnpy::npz_load(filename);

    std::map<std::string, Tensor> tensors;
    for (auto& [key, array] : npz) {
        Tensor tensor;

        std::vector<int64_t> shape;
        for (size_t i = 0; i < array.shape.size(); i++) {
            shape.push_back(static_cast<int64_t>(array.shape[i]));
        }

        if (array.word_size == 4) {
            tensor = Tensor::from_data(array.data<float>(), shape).clone();
        } else if (array.word_size == 1) {
            tensor = Tensor::from_data(array.data<uint8_t>(), shape).clone();
        } else {
            return Err<TensorMap>(PtensorError::INVALID_ARGUMENT, "Unsupported data type");
        }
        tensors.try_emplace(key, tensor);
    }

    return Ok<TensorMap>(std::move(tensors));
}
}  // namespace p10::io
