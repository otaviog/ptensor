
#include <cstring>
#include <string>
#include <vector>

#include <emscripten/val.h>
#include <ptensor/tensor.hpp>
#include "ptensor/p10_error.hpp"
#include <ptensor/tensor_print.hpp>
#include "wasm_shape.hpp"

using namespace emscripten;

class WasmTensor: public p10::Tensor {
  public:
    WasmTensor() = default;

    WasmTensor(p10::Tensor&& tensor) : p10::Tensor(std::move(tensor)) {}

    p10::P10Error fromData(const WasmShape& shape, const p10::Dtype& dtype, uintptr_t dataPtr) {
        uint8_t* data = reinterpret_cast<uint8_t*>(dataPtr);
        P10_RETURN_IF_ERROR(create(shape, dtype));

        size_t byte_size = size_bytes();
        std::memcpy(as_bytes().data(), data, byte_size);
        return p10::P10Error::Ok;
    }

    static WasmTensor* zeros(const WasmShape& shape, const p10::Dtype& dtype) {
        auto result = p10::Tensor::zeros(shape, p10::TensorOptions().dtype(dtype));
        if (result.is_error()) {
            throw std::runtime_error(result.err().to_string());
        }
        return new WasmTensor(std::move(result.unwrap()));
    }

    std::string toString() const {
        return p10::to_string(*this);
    }

    WasmShape getShape() const {        
        return WasmShape(shape());
    }

    std::string getDtype() const {
        return p10::to_string(dtype());
    }

    size_t getSize() const {
        return size();
    }
};
