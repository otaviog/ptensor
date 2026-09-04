#include "tlog.hpp"

#include <cstdlib>
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

}  // namespace

void log(const std::string& entry, const Tensor& tensor) {
    // Session serializes its own calls, no lock needed here.
    if (auto err = session().log_sync(entry, tensor); err.is_error()) {
        LOGGER.error("Could not log '{}'. {}", entry, err);
    }
}

namespace {

    struct SessionHolder {
        SessionHolder() {
            const char* address = std::getenv(ADDRESS_ENV_VAR);
            const std::string endpoint = address != nullptr ? address : DEFAULT_ADDRESS;

            if (auto err = session.start(endpoint); err.is_error()) {
                LOGGER.error("Could not start the session on {}. {}", endpoint, err);
            } else {
                LOGGER.info("Logging tensors to {}", endpoint);
            }
        }

        Session session;
    };

    Session& session() {
        static SessionHolder holder;
        return holder.session;
    }

}  // namespace

}  // namespace p10::tlog
