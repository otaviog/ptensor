#pragma once

#include <ptensor/infer/infer.hpp>

namespace p10::infer {

inline IInfer* unwrap_infer(P10Infer handle) {
    assert(handle != nullptr && "Null P10Infer handle");
    return reinterpret_cast<IInfer*>(handle);
}

inline P10Infer wrap_infer(IInfer* ptr) {
    return reinterpret_cast<P10Infer>(ptr);
}

}  // namespace p10::infer
