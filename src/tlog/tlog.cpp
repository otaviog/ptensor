#include "tlog.hpp"

#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "logging.hpp"
#include "session.hpp"

namespace p10::tlog {
namespace {

    constexpr const char* DEFAULT_ADDRESS = "localhost:4449";
    constexpr const char* ADDRESS_ENV_VAR = "PTENSOR_TLOG_ADDRESS";

    // Starts the session on construction. A holder rather than a factory
    // because a Session cannot be moved out of one.
    struct SessionHolder;

    // The process wide session, started on the first log call.
    Session& session();

    // The session for one explicit address, started on its first use.
    Session& session_for(const std::string& address);

}  // namespace

void log(const std::string& entry, const Tensor& tensor) {
    // Session serializes its own calls, no lock needed here.
    if (auto err = session().log_sync(entry, tensor); err.is_error()) {
        LOGGER.error("Could not log '{}'. {}", entry, err);
    }
}

void log_to(const char* address, const char* entry, const Tensor& tensor) {
    if (address == nullptr || entry == nullptr) {
        LOGGER.error("log_to needs both an address and an entry name.");
        return;
    }
    if (auto err = session_for(address).log_sync(entry, tensor); err.is_error()) {
        LOGGER.error("Could not log '{}' to {}. {}", entry, address, err);
    }
}

namespace {

    struct SessionHolder {
        explicit SessionHolder(const std::string& endpoint) {
            if (auto err = session.start(endpoint); err.is_error()) {
                LOGGER.error("Could not start the session on {}. {}", endpoint, err);
            } else {
                LOGGER.info("Logging tensors to {}", endpoint);
            }
        }

        Session session;
    };

    Session& session() {
        static SessionHolder holder([] {
            const char* address = std::getenv(ADDRESS_ENV_VAR);
            return address != nullptr ? std::string(address) : std::string(DEFAULT_ADDRESS);
        }());
        return holder.session;
    }

    Session& session_for(const std::string& address) {
        // Held for the life of the process: a caller that logs to an address
        // once usually logs to it again, and a Session is one socket.
        static std::mutex lock;
        static std::map<std::string, std::unique_ptr<SessionHolder>> sessions;

        const std::lock_guard<std::mutex> guard(lock);
        auto& holder = sessions[address];
        if (holder == nullptr) {
            holder = std::make_unique<SessionHolder>(address);
        }
        return holder->session;
    }

}  // namespace

}  // namespace p10::tlog
