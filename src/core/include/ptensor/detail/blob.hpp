#pragma once

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <utility>

#include <ptensor/config.h>

#include "../device.hpp"
#include "../p10_result.hpp"

namespace p10 {

/// Reference-counted storage handle.
///
/// A blob owns (or borrows) a raw memory buffer. Ownership is shared: copies and
/// views of a blob keep the underlying buffer alive until the last handle is
/// dropped. This lets tensor views (e.g. slices) safely outlive the tensor they
/// were created from.
///
/// `const` on a blob is shallow: a `const Blob` cannot be reseated, but it can
/// still hand out a writable view of the same buffer. The constness of the
/// pointed-to memory is not tracked here (matching the `Tensor` handle model).
class Blob {
  public:
    static Blob allocate(size_t size);

    Blob() = default;

    /// Creates a blob from a raw pointer.
    ///
    /// * If `dealloc` is set, the blob owns the buffer and runs `dealloc` when
    ///   the last handle is dropped.
    /// * If `dealloc` is empty, the buffer is borrowed and never freed.
    Blob(
        void* data,
        const Device& device,
        std::optional<std::function<void(void*)>> dealloc = std::nullopt
    ) :
        data_ {make_data_ptr(data, std::move(dealloc))},
        device_ {device} {}

    Blob(const Blob&) = default;
    Blob& operator=(const Blob&) = default;
    Blob(Blob&&) noexcept = default;
    Blob& operator=(Blob&&) noexcept = default;
    ~Blob() = default;

    /// Returns a blob sharing this blob's storage, offset by `offset` bytes.
    ///
    /// The returned view shares ownership: it keeps the underlying buffer alive
    /// even if the originating blob is dropped.
    Blob view(size_t offset = 0) const {
        // Aliasing shared_ptr: shares the refcount of `data_` while pointing
        // into the middle of the buffer.
        return Blob(
            std::shared_ptr<void>(data_, static_cast<uint8_t*>(data_.get()) + offset),
            device_
        );
    }

    template<typename scalar_t>
    scalar_t* data() {
        return static_cast<scalar_t*>(data_.get());
    }

    template<typename scalar_t>
    const scalar_t* data() const {
        return static_cast<const scalar_t*>(data_.get());
    }

    Device device() const {
        return device_;
    }

    /// Returns a blob sharing this blob's storage.
    P10Result<Blob> clone() const {
        return Ok(Blob(data_, device_));
    }

    Blob copy(size_t size) const {
        Blob new_blob = Blob::allocate(size);
        std::memcpy(new_blob.data<std::byte>(), data<std::byte>(), size);
        return new_blob;
    }

    bool is_aligned(size_t alignment) const {
        return (reinterpret_cast<uintptr_t>(data_.get()) % alignment) == 0;
    }

  private:
    Blob(std::shared_ptr<void> data, const Device& device) :
        data_ {std::move(data)},
        device_ {device} {}

    static std::shared_ptr<void>
    make_data_ptr(void* data, std::optional<std::function<void(void*)>> dealloc) {
        if (data == nullptr) {
            return nullptr;
        }
        if (dealloc.has_value()) {
            return std::shared_ptr<void>(data, std::move(dealloc.value()));
        }
        // Borrowed buffer: shares a refcount but never frees the memory.
        return std::shared_ptr<void>(data, [](void*) {});
    }

    std::shared_ptr<void> data_;
    Device device_;
};
}  // namespace p10
