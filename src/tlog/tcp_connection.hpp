#pragma once

#include <cstddef>
#include <span>
#include <string>

#include "connection.hpp"

namespace p10::tlog {

/// Blocking TCP client connection. POSIX only for now.
class TcpConnection: public IConnection {
    static constexpr int INVALID_SOCKET = -1;

  public:
    TcpConnection() = default;

    ~TcpConnection() override {
        close();
    }

    // Owns a socket descriptor, so copying it would close the same fd twice.
    // The base already deletes these, they are repeated here to say so.
    TcpConnection(const TcpConnection&) = delete;
    TcpConnection(TcpConnection&&) = delete;
    TcpConnection& operator=(const TcpConnection&) = delete;
    TcpConnection& operator=(TcpConnection&&) = delete;

    /// Connects to `address`, given as "host:port". The host may be a name or
    /// a numeric IPv4/IPv6 address.
    P10Error connect(const std::string& address) override;

    void close() override;

    P10Error send(std::span<const std::byte> data) const override;

    // The overriding declaration above would otherwise hide the string overload.
    using IConnection::send;

  private:
    int socket_ = INVALID_SOCKET;
};
}  // namespace p10::tlog
