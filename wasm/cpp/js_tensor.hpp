
#include <cstring>
#include <string>
#include <vector>

#include <ptensor/p10_error.hpp>
#include <ptensor/tensor.hpp>
#include <ptensor/tensor_print.hpp>

#include "common.hpp"
#include "js_error.hpp"
#include "js_shape.hpp"
#include "ptensor/dtype.hpp"

class JsTensor: public p10::Tensor {
  public:
    JsTensor() = default;

    JsTensor(p10::Tensor&& tensor) : p10::Tensor(std::move(tensor)) {}

    static JsTensor* fromData(const JsShape& shape, const p10::Dtype& dtype, uintptr_t dataPtr) {
        uint8_t* data = reinterpret_cast<uint8_t*>(dataPtr);
        // In JS we allocate with malloc, here so we free with free.
        auto tensor = p10::Tensor::from_data(data, shape, p10::TensorOptions().dtype(dtype), free);
        return new JsTensor(std::move(tensor));
    }

    static JsTensor* zeros(const JsShape& shape, const p10::Dtype& dtype) {
        auto result = p10::Tensor::zeros(shape, p10::TensorOptions().dtype(dtype));
        if (result.is_error()) {
            throw std::runtime_error(result.err().to_string());
        }
        return new JsTensor(std::move(result.unwrap()));
    }

    std::string toString() const {
        return p10::to_string(*this);
    }

    JsShape getShape() const {
        return JsShape(shape());
    }

    p10::Dtype getDtype() const {
        return dtype();
    }

    size_t getSize() const {
        return size();
    }
};
