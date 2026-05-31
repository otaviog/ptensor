#include <ng-log/logging.h>
#include <p10_internal/log/log.hpp>

namespace p10::log::detail {

void log_info(std::string_view msg) {
    LOG(INFO) << msg;
}

void log_error(std::string_view msg) {
    LOG(ERROR) << msg;
}

void log_debug(std::string_view msg) {
    DLOG(INFO) << msg;
}

}  // namespace p10::log::detail
