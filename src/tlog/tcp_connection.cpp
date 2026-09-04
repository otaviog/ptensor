#include "tcp_connection.hpp"

#include <cerrno>
#include <cstring>
#include <utility>

#include <netdb.h>
#include <ptensor/p10_error.hpp>
#include <ptensor/p10_result.hpp>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "logging.hpp"

namespace p10::tlog {

namespace {
    // Splits "host:port" into its two halves. The port is kept as text
    // because that is what getaddrinfo() expects.
    P10Result<std::pair<std::string, std::string>> split_address(const std::string& address);
}  // namespace

P10Error TcpConnection::connect(const std::string& address) {
    if (socket_ != INVALID_SOCKET) {
        return P10Error::IoError << "Socket already connected";
    }

    auto parts = split_address(address);
    P10_RETURN_IF_ERROR(parts.error());
    const auto& [host, port] = parts.unwrap();

    const addrinfo hints {.ai_flags = AI_NUMERICSERV, .ai_socktype = SOCK_STREAM};
    addrinfo* resolved = nullptr;
    if (const int err = ::getaddrinfo(host.c_str(), port.c_str(), &hints, &resolved); err != 0) {
        return P10Error::InvalidArgument
            << "Could not resolve '" + address + "': " + ::gai_strerror(err);
    }

    P10Error last_error = P10Error::IoError << "Could not connect to " + address;
    for (const addrinfo* entry = resolved; entry != nullptr; entry = entry->ai_next) {
        const int handle = ::socket(entry->ai_family, entry->ai_socktype, entry->ai_protocol);
        if (handle < 0) {
            last_error = P10Error::current_os_error();
            continue;
        }

#ifdef SO_NOSIGPIPE
        // macOS/BSD: ask the socket not to raise SIGPIPE on a broken peer.
        const int enable = 1;
        ::setsockopt(handle, SOL_SOCKET, SO_NOSIGPIPE, &enable, sizeof(enable));
#endif

        if (::connect(handle, entry->ai_addr, entry->ai_addrlen) == 0) {
            socket_ = handle;
            break;
        }

        last_error = P10Error::current_os_error();
        ::close(handle);
    }

    ::freeaddrinfo(resolved);

    if (socket_ == INVALID_SOCKET) {
        return last_error;
    }

    return P10Error::Ok;
}

void TcpConnection::close() {
    if (socket_ == INVALID_SOCKET) {
        return;
    }

    ::close(socket_);
    socket_ = INVALID_SOCKET;
}

P10Error TcpConnection::send(std::span<const std::byte> data) const {
    if (socket_ == INVALID_SOCKET) {
        return P10Error::IoError << "Socket is not connected";
    }

#ifdef MSG_NOSIGNAL
    // Linux: the per call equivalent of SO_NOSIGPIPE.
    constexpr int FLAGS = MSG_NOSIGNAL;
#else
    constexpr int FLAGS = 0;
#endif

    size_t written = 0;
    while (written < data.size()) {
        const ssize_t sent_len =
            ::send(socket_, data.data() + written, data.size() - written, FLAGS);
        if (sent_len < 0) {
            if (errno == EINTR) {
                continue;
            }
            const auto err = P10Error::current_os_error();
            LOGGER.error("Socket send: {}", err.to_string());
            return err;
        }
        written += static_cast<size_t>(sent_len);
    }

    return P10Error::Ok;
}

namespace {
    P10Result<std::pair<std::string, std::string>> split_address(const std::string& address) {
        const auto found_colon = address.find_last_of(':');
        if (found_colon == std::string::npos || found_colon + 1 == address.size()
            || found_colon == 0) {
            return Err(P10Error::InvalidArgument, "Address must be in the format 'host:port'");
        }

        std::string host = address.substr(0, found_colon);
        std::string port = address.substr(found_colon + 1);

        if (port.find_first_not_of("0123456789") != std::string::npos) {
            return Err(P10Error::InvalidArgument, "Invalid port number: " + port);
        }
        if (std::stol(port) > 65535) {
            return Err(P10Error::InvalidArgument, "Port number out of range: " + port);
        }

        return Ok(std::pair {std::move(host), std::move(port)});
    }
}  // namespace
}  // namespace p10::tlog
