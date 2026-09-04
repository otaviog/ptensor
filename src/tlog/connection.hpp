#pragma once

#include <cstddef>
#include <span>
#include <string>

#include <ptensor/p10_error.hpp>

namespace p10::tlog {

class IConnection {
  public:
    virtual ~IConnection() = default;
    IConnection(const IConnection&) = delete;
    IConnection(const IConnection&&) = delete;
    IConnection& operator=(const IConnection&) = delete;
    IConnection& operator=(const IConnection&&) = delete;

    virtual P10Error connect(const std::string& address) = 0;
    virtual void close() = 0;
    virtual P10Error send(std::span<const std::byte> data) const = 0;

    P10Error send(const std::string& data) const {
        return send(std::as_bytes(std::span(data)));
    }

  protected:
    IConnection() = default;
};
}  // namespace p10::tlog
