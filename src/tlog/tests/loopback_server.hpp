#pragma once

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace p10::tlog::testing {

/// Minimal loopback TCP server for the connection tests: binds an ephemeral
/// port, accepts one client and reads until the peer closes.
class LoopbackServer {
  public:
    LoopbackServer() {
        listen_socket_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_socket_ < 0) {
            return;
        }

        const int enable = 1;
        ::setsockopt(listen_socket_, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));

        // Bound waits, so a test that never connects cannot hang the worker.
        const timeval timeout {.tv_sec = 5, .tv_usec = 0};
        ::setsockopt(listen_socket_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

        sockaddr_in addr {};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = 0;  // Let the kernel pick a free port.

        if (::bind(listen_socket_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0
            || ::listen(listen_socket_, 1) < 0) {
            stop();
            return;
        }

        socklen_t addr_len = sizeof(addr);
        if (::getsockname(listen_socket_, reinterpret_cast<sockaddr*>(&addr), &addr_len) < 0) {
            stop();
            return;
        }
        port_ = ntohs(addr.sin_port);

        worker_ = std::thread([this] { serve(); });
    }

    ~LoopbackServer() {
        stop();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    LoopbackServer(const LoopbackServer&) = delete;
    LoopbackServer& operator=(const LoopbackServer&) = delete;

    bool is_listening() const {
        return port_ != 0;
    }

    std::string address() const {
        return "127.0.0.1:" + std::to_string(port_);
    }

    /// Everything the client sent. Call it once the client has closed its end:
    /// it waits for the worker to drain the socket rather than cutting it off.
    std::string received() {
        if (worker_.joinable()) {
            worker_.join();
        }
        stop();

        const std::lock_guard<std::mutex> lock(mutex_);
        return received_;
    }

    /// Waits until at least `count` bytes arrived, for clients that keep the
    /// connection open, and returns what has been read so far.
    std::string
    wait_for(size_t count, std::chrono::milliseconds timeout = std::chrono::seconds(2)) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            {
                const std::lock_guard<std::mutex> lock(mutex_);
                if (received_.size() >= count) {
                    return received_;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        const std::lock_guard<std::mutex> lock(mutex_);
        return received_;
    }

  private:
    void serve() {
        const int client = ::accept(listen_socket_, nullptr, nullptr);
        if (client < 0) {
            return;
        }
        client_socket_ = client;

        const timeval timeout {.tv_sec = 5, .tv_usec = 0};
        ::setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

        char buffer[1024];
        while (true) {
            const ssize_t read_len = ::recv(client, buffer, sizeof(buffer), 0);
            if (read_len <= 0) {
                break;
            }
            const std::lock_guard<std::mutex> lock(mutex_);
            received_.append(buffer, static_cast<size_t>(read_len));
        }

        client_socket_ = -1;
        ::close(client);
    }

    /// Unblocks the worker: the client may keep its end open forever, so the
    /// accepted socket is shut down as well.
    void stop() {
        if (const int client = client_socket_.exchange(-1); client >= 0) {
            ::shutdown(client, SHUT_RDWR);
        }

        if (listen_socket_ >= 0) {
            ::shutdown(listen_socket_, SHUT_RDWR);
            ::close(listen_socket_);
            listen_socket_ = -1;
        }
    }

    std::thread worker_;
    std::mutex mutex_;
    std::string received_;
    std::atomic<int> client_socket_ = -1;
    int listen_socket_ = -1;
    uint16_t port_ = 0;
};

}  // namespace p10::tlog::tests
