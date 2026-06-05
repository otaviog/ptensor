#include "detail/blob.hpp"

namespace p10 {

Blob Blob::allocate(size_t size) {
    auto* memory = new uint8_t[size];
    if (memory == nullptr) {
        throw std::bad_alloc();
    }
    return Blob(static_cast<void*>(memory), Device(Device::Cpu), [](void* data) {
        delete[] static_cast<uint8_t*>(data);
    });
}

}  // namespace p10
