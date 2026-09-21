#include "session.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <functional>
#include <mutex>
#include <random>

#include <ptensor/json.hpp>
#include <ptensor/p10_error.hpp>
#include <ptensor/tensor.hpp>
#include <uuid.h>

#include "tcp_connection.hpp"

namespace p10::tlog {
namespace {

    // Returns a version-4 UUID string, used to identify a session on the server.
    std::string generate_random_uuid();

}  // namespace

Session::Session() : conn_(std::make_unique<TcpConnection>()) {}

P10Error Session::start(const std::string& address) {
    const std::lock_guard<SpinLock> guard(lock_);

    const auto session_id = generate_random_uuid();
    P10_RETURN_IF_ERROR(conn_->connect(address));
    // Newline delimited, like the entries: the server reads the stream a line
    // at a time, so a handshake without one glues onto the first entry.
    return conn_->send(std::format("{{\"sessionId\":\"{}\"}}\n", session_id));
}

P10Error Session::log_sync(const std::string& name, const Tensor& tensor) {
    const std::lock_guard<SpinLock> guard(lock_);

    // The buffer is a member so repeated logs reuse its capacity. The entry is
    // assembled by hand rather than formatted: the tensor is most of the line
    // and append_json writes it straight into the buffer.
    json_buffer_.clear();
    json_buffer_ += R"({"name":")";
    json_buffer_ += name;
    json_buffer_ += R"(","tensor":)";
    append_json(json_stage_.get_encoder(tensor, JsonEncodeMode::Base64Compressed), json_buffer_);
    json_buffer_ += "}\n";

    return conn_->send(json_buffer_);
}

namespace {

    std::string generate_random_uuid() {
        std::random_device rd;
        std::array<int, std::mt19937::state_size> seed_data {};
        std::ranges::generate(seed_data, std::ref(rd));
        std::seed_seq seq(seed_data.begin(), seed_data.end());
        std::mt19937 engine(seq);

        uuids::uuid_random_generator generator {engine};
        return uuids::to_string(generator());
    }

}  // namespace

}  // namespace p10::tlog
