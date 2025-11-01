#include "numpy.hpp"

#include <algorithm>
#include <cstdint>

#include <cnpy.h>

#include "ptensor/p10_error.hpp"
#include "ptensor/p10_result.hpp"

namespace p10::io {
using TensorMap = std::map<std::string, Tensor>;

P10Error save_npz(const std::string& filename, const TensorMap& tensors) {
    std::string mode = "w";
    for (const auto& [key_name, tensor] : tensors) {
        std::vector<size_t> shape;

        std::transform(
            tensor.shape().begin(),
            tensor.shape().end(),
            std::back_inserter(shape),
            [](const int64_t& dim) { return static_cast<size_t>(dim); }
        );
        tensor.visit([&](auto span) {
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4267 4996 4310)
#endif
            cnpy::npz_save(filename, key_name, span.data(), shape, mode);
#ifdef _MSC_VER
    #pragma warning(pop)
#endif
        });
        mode = "a";
    }

    return P10Error::Ok;
}

P10Result<TensorMap> load_npz(const std::string& filename) {
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4267)
#endif
    try {
        cnpy::npz_t npz = cnpy::npz_load(filename);

#ifdef _MSC_VER
    #pragma warning(pop)
#endif

        std::map<std::string, Tensor> tensors;
        for (auto& [key, array] : npz) {
            Tensor tensor;

            std::vector<int64_t> shape;
            for (size_t i = 0; i < array.shape.size(); i++) {
                shape.push_back(static_cast<int64_t>(array.shape[i]));
            }

            auto p10_shape = make_shape(shape);
            if (!p10_shape.is_ok()) {
                return Err(p10_shape.unwrap_err());
            }

            if (array.word_size == 4) {
                tensor =
                    Tensor::from_data(array.data<float>(), p10_shape.unwrap()).clone().unwrap();
            } else if (array.word_size == 1) {
                tensor =
                    Tensor::from_data(array.data<uint8_t>(), p10_shape.unwrap()).clone().unwrap();
            } else {
                return Err(P10Error::InvalidArgument, "Unsupported data type");
            }
            tensors.try_emplace(key, std::move(tensor));
        }

        return Ok<TensorMap>(std::move(tensors));
    } catch (const std::exception& e) {
        return Err(P10Error::InvalidArgument, e.what());
    }
}
}  // namespace p10::io
