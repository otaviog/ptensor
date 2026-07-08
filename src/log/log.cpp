#include <filesystem>
#include <memory>

#include <p10_internal/log/log.hpp>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include "ptensor/initialize.hpp"

namespace p10::log::detail {
namespace {
    std::shared_ptr<spdlog::logger>& logger();
}

void log_info(std::string_view msg) {
    logger()->info("{}", msg);
}

void log_error(std::string_view msg) {
    logger()->error("{}", msg);
}

void log_debug(std::string_view msg) {
    logger()->debug("{}", msg);
}

namespace {
    std::shared_ptr<spdlog::logger>& logger() {
        static std::shared_ptr<spdlog::logger> instance = [] {
            std::filesystem::path const log_dir(get_log_directory());
            if (!std::filesystem::exists(log_dir)) {
                std::filesystem::create_directory(log_dir);
            }

            auto log = spdlog::basic_logger_mt("ptensor", (log_dir / "ptensor.log").string());
#ifdef NDEBUG
            log->set_level(spdlog::level::info);
#else
            log->set_level(spdlog::level::debug);
#endif
            return log;
        }();
        return instance;
    }
}  // namespace
}  // namespace p10::log::detail
