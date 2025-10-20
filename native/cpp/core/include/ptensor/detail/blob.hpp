#pragma once

#include <cstring>
#include <functional>
#include <optional>

#include <ptensor/config.h>

#include "../device.hpp"
#include "../p10_result.hpp"

namespace p10 {
class Blob {
  public:
    static Blob allocate(size_t size);

    Blob() = default;

    Blob(
        void* data,
        const Device& device,
        std::optional<std::function<void(void*)>> dealloc = std::nullopt
    ) :
        data_ {data},
        dealloc_ {dealloc},
        device_ {device} {}

    Blob(const Blob& other) = delete;

    Blob& operator=(const Blob& other) = delete;

    Blob(Blob&& other) noexcept;

    Blob& operator=(Blob&& other) noexcept;

    ~Blob();

    Blob view() {
        return Blob(data_, device_);
    }

    template<typename scalar_t>
    scalar_t* data() {
        return static_cast<scalar_t*>(data_);
    }

    template<typename scalar_t>
    const scalar_t* data() const {
        return static_cast<scalar_t*>(data_);
    }

    Device device() const {
        return device_;
    }

    P10Result<Blob> clone() const {
        if (dealloc_.has_value()) {
            return Err(P10Error::InvalidOperation, "Cannot clone a Blob that owns its data");
        }
        return Ok(Blob(data_, device_));
    }

    Blob copy(size_t size) const {
        Blob new_blob = Blob::allocate(size);
        std::memcpy(new_blob.data<uint8_t>(), data<uint8_t>(), size);
        return new_blob;
    }

  private:
    void* data_ = nullptr;
    std::optional<std::function<void(void*)>> dealloc_;
    Device device_;
};
}  // namespace p10
