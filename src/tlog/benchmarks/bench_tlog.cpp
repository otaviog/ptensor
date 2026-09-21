#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <format>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <benchmark/benchmark.h>
#include <netinet/in.h>
#include <ptensor/json.hpp>
#include <ptensor/tensor.hpp>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "base64.hpp"
#include "compress_staging.hpp"
#include "connection.hpp"
#include "session.hpp"

namespace p10::tlog {
namespace {

    // ---------------------------------------------------------------- payloads

    // Incompressible payload: worst case for the zstd stage.
    Tensor make_noise(int64_t rows, int64_t cols) {
        const std::mt19937_64 rng(42);
        return Tensor::from_random(
                   make_shape(rows, cols, 3),
                   rng,
                   TensorOptions().dtype(Dtype::Uint8),
                   0.0,
                   255.0
        )
            .unwrap();
    }

    // Flat, highly compressible payload: the best case for the zstd stage.
    Tensor make_gradient(int64_t rows, int64_t cols) {
        Tensor tensor =
            Tensor::empty(make_shape(rows, cols, 3), TensorOptions().dtype(Dtype::Uint8)).unwrap();
        const std::span<std::byte> bytes = tensor.as_bytes();
        for (size_t i = 0; i < bytes.size(); ++i) {
            bytes[i] = static_cast<std::byte>((i / 16) & 0xFF);
        }
        return tensor;
    }

    // Photo-like payload: a smooth ramp plus per-pixel noise, so zstd finds
    // some redundancy but not much. The closest of the three to a camera frame.
    Tensor make_photo(int64_t rows, int64_t cols) {
        Tensor tensor =
            Tensor::empty(make_shape(rows, cols, 3), TensorOptions().dtype(Dtype::Uint8)).unwrap();
        const std::span<std::byte> bytes = tensor.as_bytes();
        std::mt19937_64 rng(42);
        std::uniform_int_distribution<int> noise(0, 40);
        for (size_t i = 0; i < bytes.size(); ++i) {
            bytes[i] = static_cast<std::byte>(((i / 64) + noise(rng)) & 0xFF);
        }
        return tensor;
    }

    Tensor make_payload(const benchmark::State& state) {
        const int64_t rows = state.range(0);
        const int64_t cols = state.range(1);
        switch (state.range(2)) {
            case 0:
                return make_gradient(rows, cols);
            case 1:
                return make_photo(rows, cols);
            default:
                return make_noise(rows, cols);
        }
    }

    // Bytes of the tensor itself, so every case is comparable in MB/s of input.
    void report(benchmark::State& state, const Tensor& tensor) {
        state.SetBytesProcessed(
            static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(tensor.size_bytes())
        );
    }

    // ------------------------------------------------------------- connections

    /// Accepts what it is given and drops it: isolates the encode cost from the
    /// socket cost.
    class NullConnection: public IConnection {
      public:
        P10Error connect(const std::string&) override {
            return P10Error::Ok;
        }

        void close() override {}

        P10Error send(std::span<const std::byte> data) const override {
            benchmark::DoNotOptimize(data.data());
            return P10Error::Ok;
        }

        using IConnection::send;
    };

    /// Loopback TCP peer that only drains its socket. Reads in large chunks so
    /// the reader does not become the bottleneck being measured.
    class DrainServer {
      public:
        DrainServer() {
            listen_socket_ = ::socket(AF_INET, SOCK_STREAM, 0);
            const int enable = 1;
            ::setsockopt(listen_socket_, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));

            sockaddr_in addr {};
            addr.sin_family = AF_INET;
            addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            addr.sin_port = 0;
            ::bind(listen_socket_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
            ::listen(listen_socket_, 1);

            socklen_t addr_len = sizeof(addr);
            ::getsockname(listen_socket_, reinterpret_cast<sockaddr*>(&addr), &addr_len);
            port_ = ntohs(addr.sin_port);

            worker_ = std::thread([this] { serve(); });
        }

        ~DrainServer() {
            if (listen_socket_ >= 0) {
                ::shutdown(listen_socket_, SHUT_RDWR);
                ::close(listen_socket_);
                listen_socket_ = -1;
            }
            if (worker_.joinable()) {
                worker_.join();
            }
        }

        DrainServer(const DrainServer&) = delete;
        DrainServer(DrainServer&&) = delete;
        DrainServer& operator=(const DrainServer&) = delete;
        DrainServer& operator=(DrainServer&&) = delete;

        std::string address() const {
            return "127.0.0.1:" + std::to_string(port_);
        }

      private:
        void serve() {
            const int client = ::accept(listen_socket_, nullptr, nullptr);
            if (client < 0) {
                return;
            }
            std::vector<char> buffer(1 << 18);
            while (::recv(client, buffer.data(), buffer.size(), 0) > 0) {}
            ::close(client);
        }

        std::thread worker_;
        int listen_socket_ = -1;
        uint16_t port_ = 0;
    };

    /// Drains an AF_UNIX stream socket, to price the transport against TCP on
    /// loopback. The path lives in the temp dir and is unlinked on teardown.
    class UnixDrainServer {
      public:
        UnixDrainServer() {
            path_ = std::filesystem::temp_directory_path()
                / ("ptensor-bench-" + std::to_string(::getpid()) + ".sock");
            ::unlink(path_.c_str());

            listen_socket_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
            sockaddr_un addr {};
            addr.sun_family = AF_UNIX;
            std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", path_.c_str());
            ::bind(listen_socket_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
            ::listen(listen_socket_, 1);

            worker_ = std::thread([this] { serve(); });
        }

        ~UnixDrainServer() {
            if (listen_socket_ >= 0) {
                ::shutdown(listen_socket_, SHUT_RDWR);
                ::close(listen_socket_);
                listen_socket_ = -1;
            }
            if (worker_.joinable()) {
                worker_.join();
            }
            ::unlink(path_.c_str());
        }

        UnixDrainServer(const UnixDrainServer&) = delete;
        UnixDrainServer(UnixDrainServer&&) = delete;
        UnixDrainServer& operator=(const UnixDrainServer&) = delete;
        UnixDrainServer& operator=(UnixDrainServer&&) = delete;

        const std::filesystem::path& path() const {
            return path_;
        }

      private:
        void serve() {
            const int client = ::accept(listen_socket_, nullptr, nullptr);
            if (client < 0) {
                return;
            }
            std::vector<char> buffer(1 << 18);
            while (::recv(client, buffer.data(), buffer.size(), 0) > 0) {}
            ::close(client);
        }

        std::thread worker_;
        std::filesystem::path path_;
        int listen_socket_ = -1;
    };

    /// The same blocking write loop as TcpConnection, over an AF_UNIX socket.
    class UnixConnection: public IConnection {
      public:
        ~UnixConnection() override {
            close();
        }

        P10Error connect(const std::string& address) override {
            socket_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
            sockaddr_un addr {};
            addr.sun_family = AF_UNIX;
            std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", address.c_str());
            // macOS defaults an AF_UNIX send buffer to a few KiB, which makes
            // the writer block far more often than loopback TCP does. Raise it
            // so the comparison is about the transport, not the buffer size.
            const int buffer_size = 1 << 20;
            ::setsockopt(socket_, SOL_SOCKET, SO_SNDBUF, &buffer_size, sizeof(buffer_size));

            if (::connect(socket_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
                return P10Error::current_os_error();
            }
            return P10Error::Ok;
        }

        void close() override {
            if (socket_ >= 0) {
                ::close(socket_);
                socket_ = -1;
            }
        }

        P10Error send(std::span<const std::byte> data) const override {
            size_t written = 0;
            while (written < data.size()) {
                const ssize_t sent =
                    ::send(socket_, data.data() + written, data.size() - written, 0);
                if (sent < 0) {
                    if (errno == EINTR) {
                        continue;
                    }
                    return P10Error::current_os_error();
                }
                written += static_cast<size_t>(sent);
            }
            return P10Error::Ok;
        }

        using IConnection::send;

      private:
        int socket_ = -1;
    };

    // ------------------------------------------------------------- benchmarks

    // Whole client path minus the socket: JSON header, zstd, base64, buffer.
    void BM_Session_LogSync_Null(benchmark::State& state) {
        const Tensor tensor = make_payload(state);
        Session session(std::make_unique<NullConnection>());
        static_cast<void>(session.start("null:0"));

        for (auto _ : state) {
            P10Error err = session.log_sync("bench", tensor);
            benchmark::DoNotOptimize(err);
            benchmark::ClobberMemory();
        }
        report(state, tensor);
    }

    // Same path over a real loopback socket: the delta against the null case is
    // what write() plus the kernel copy costs.
    void BM_Session_LogSync_Loopback(benchmark::State& state) {
        const Tensor tensor = make_payload(state);
        const DrainServer server;
        Session session;
        if (session.start(server.address()).is_error()) {
            state.SkipWithError("could not connect to the loopback server");
            return;
        }

        for (auto _ : state) {
            P10Error err = session.log_sync("bench", tensor);
            benchmark::DoNotOptimize(err);
            benchmark::ClobberMemory();
        }
        report(state, tensor);
    }

    // Same path over an AF_UNIX stream socket, to see whether dropping the
    // loopback TCP stack is worth anything at these payload sizes.
    void BM_Session_LogSync_UnixSocket(benchmark::State& state) {
        const Tensor tensor = make_payload(state);
        const UnixDrainServer server;
        Session session(std::make_unique<UnixConnection>());
        if (session.start(server.path().string()).is_error()) {
            state.SkipWithError("could not connect to the unix socket server");
            return;
        }

        for (auto _ : state) {
            P10Error err = session.log_sync("bench", tensor);
            benchmark::DoNotOptimize(err);
            benchmark::ClobberMemory();
        }
        report(state, tensor);
    }

    // Encoding only, both modes, so the zstd stage can be priced on its own.
    // Mirrors Session::log_sync: the entry is appended, not formatted.
    void run_encode(benchmark::State& state, JsonEncodeMode mode) {
        const Tensor tensor = make_payload(state);
        JsonStaging staging;
        std::string buffer;

        for (auto _ : state) {
            buffer.clear();
            buffer += R"({"name":"bench","tensor":)";
            append_json(staging.get_encoder(tensor, mode), buffer);
            buffer += "}\n";
            benchmark::DoNotOptimize(buffer.data());
            benchmark::ClobberMemory();
        }
        // What actually goes on the wire, so the ratio is visible next to the time.
        state.counters["wire_bytes"] = static_cast<double>(buffer.size());
        report(state, tensor);
    }

    void BM_Encode_Compressed(benchmark::State& state) {
        run_encode(state, JsonEncodeMode::Base64Compressed);
    }

    void BM_Encode_Plain(benchmark::State& state) {
        run_encode(state, JsonEncodeMode::Base64);
    }

    // The same object through std::formatter<Json>, which is what the public
    // `std::format("{}", encoder)` spelling costs: it encodes into a buffer and
    // then copies that buffer out one character at a time.
    void BM_Encode_ViaFormatter(benchmark::State& state) {
        const Tensor tensor = make_payload(state);
        JsonStaging staging;
        std::string buffer;

        for (auto _ : state) {
            buffer.clear();
            std::format_to(
                std::back_inserter(buffer),
                "{{\"name\":\"bench\",\"tensor\":{}}}\n",
                staging.get_encoder(tensor, JsonEncodeMode::Base64Compressed)
            );
            benchmark::DoNotOptimize(buffer.data());
            benchmark::ClobberMemory();
        }
        report(state, tensor);
    }

    // zstd level 1 alone, on a reused staging buffer.
    void BM_Zstd_Compress(benchmark::State& state) {
        const Tensor tensor = make_payload(state);
        detail::CompressStaging staging;

        size_t compressed_size = 0;
        for (auto _ : state) {
            const std::span<const std::byte> out = staging.compress(tensor.as_bytes());
            compressed_size = out.size();
            benchmark::DoNotOptimize(out.data());
            benchmark::ClobberMemory();
        }
        state.counters["zstd_bytes"] = static_cast<double>(compressed_size);
        report(state, tensor);
    }

    // base64 as the encoder does it today: one character at a time through the
    // std::format output iterator.
    void BM_Base64_ViaFormat(benchmark::State& state) {
        const Tensor tensor = make_payload(state);
        std::string buffer;

        for (auto _ : state) {
            buffer.clear();
            std::format_to(std::back_inserter(buffer), "{}", Base64(tensor.as_bytes()));
            benchmark::DoNotOptimize(buffer.data());
            benchmark::ClobberMemory();
        }
        report(state, tensor);
    }

    // What the send path uses: the same scalar encoder writing into a sized
    // buffer. The gap against BM_Base64_ViaFormat is the iterator's overhead.
    void BM_Base64_ToBuffer(benchmark::State& state) {
        const Tensor tensor = make_payload(state);
        std::string buffer;

        for (auto _ : state) {
            buffer.clear();
            base64_append(tensor.as_bytes(), buffer);
            benchmark::DoNotOptimize(buffer.data());
            benchmark::ClobberMemory();
        }
        report(state, tensor);
    }

    // rows, cols and the payload kind: 0 = flat gradient (very compressible),
    // 1 = photo-like (some redundancy), 2 = noise (incompressible). 480p and
    // 720p RGB frames, the sizes a viewer session actually sends.
#define PTENSOR_TLOG_SIZES(bench) \
    BENCHMARK(bench) \
        ->Args({480, 640, 0}) \
        ->Args({480, 640, 1}) \
        ->Args({480, 640, 2}) \
        ->Args({720, 1280, 1}) \
        ->Unit(benchmark::kMicrosecond)

    PTENSOR_TLOG_SIZES(BM_Session_LogSync_Null);
    PTENSOR_TLOG_SIZES(BM_Session_LogSync_Loopback);
    PTENSOR_TLOG_SIZES(BM_Session_LogSync_UnixSocket);
    PTENSOR_TLOG_SIZES(BM_Encode_Compressed);
    PTENSOR_TLOG_SIZES(BM_Encode_Plain);
    PTENSOR_TLOG_SIZES(BM_Encode_ViaFormatter);
    PTENSOR_TLOG_SIZES(BM_Zstd_Compress);
    PTENSOR_TLOG_SIZES(BM_Base64_ViaFormat);
    PTENSOR_TLOG_SIZES(BM_Base64_ToBuffer);

}  // namespace
}  // namespace p10::tlog

BENCHMARK_MAIN();
