#pragma once

#include <memory>
#include <string>

#include <ptensor/p10_error.hpp>
#include <ptensor/json.hpp>

#include "connection.hpp"
#include "spin_lock.hpp"

namespace p10 {
class Tensor;
}

namespace p10::tlog {

/// Client side of a tensor log: opens a connection, announces a session id and
/// streams newline delimited JSON entries, one per logged tensor.
///
/// The calls are serialized, so a session can be shared by several threads.
/// Holding a lock makes it neither copyable nor movable.
class Session {
  public:
    Session();

    explicit Session(std::unique_ptr<IConnection>&& conn) : conn_(std::move(conn)) {}

    /// Connects to `address` ("host:port") and sends the session handshake.
    P10Error start(const std::string& address);

    /// Sends `tensor` under `name`, blocking until the payload is written.
    P10Error log_sync(const std::string& name, const Tensor& tensor);

  private:
    SpinLock lock_;
    std::unique_ptr<IConnection> conn_;
    std::string json_buffer_;
    p10::JsonStaging json_stage_;
};
}  // namespace p10::tlog
