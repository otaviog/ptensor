#include "detail/blob.hpp"

namespace p10 {

Blob Blob::allocate(size_t size) {
    return Blob(static_cast<void*>(new uint8_t[size]), Device(Device::Cpu), [](void* data) {
        delete[] (uint8_t*)data;
    });
}

Blob::Blob(Blob&& other) noexcept {
    data_ = other.data_;
    dealloc_ = other.dealloc_;
    device_ = other.device_;
    other.data_ = nullptr;
    other.dealloc_ = std::nullopt;
    other.device_ = Device(Device::Cpu);
}

Blob& Blob::operator=(Blob&& other) noexcept {
    data_ = other.data_;
    dealloc_ = other.dealloc_;
    device_ = other.device_;
    other.data_ = nullptr;
    other.dealloc_ = std::nullopt;
    other.device_ = Device(Device::Cpu);
    return *this;
}

Blob::~Blob() {
    if (data_ != nullptr && dealloc_.has_value()) {
        dealloc_.value()(data_);
        data_ = nullptr;
    }
}

}  // namespace p10
