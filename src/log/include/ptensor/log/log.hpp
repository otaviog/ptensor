#pragma once

#include <format>
#include <string>
#include <utility>
#include <ng-log/logging.h>

#include <ptensor/p10_result.hpp>

namespace p10::log {

struct ScopedLogger {
    explicit ScopedLogger(std::string scope) noexcept : scope(std::move(scope)) {}

    template <typename... Args>
    void info(std::format_string<Args...> fmt, Args&&... args) const {
        LOG(INFO) << scope << ": " << std::format(fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void error(std::format_string<Args...> fmt, Args&&... args) const {
        LOG(ERROR) << scope << ": " << std::format(fmt, std::forward<Args>(args)...);
    }

    void error(const std::exception& exeception) const {
        LOG(ERROR) << scope << ": Exception throw" << exeception.what();
    }

    template <typename... Args>
    void debug(std::format_string<Args...> fmt, Args&&... args) const {
        DLOG(INFO) << scope << ": " << std::format(fmt, std::forward<Args>(args)...);
    }

    std::string scope;
};

inline ScopedLogger scope(std::string scope) {
    return ScopedLogger {std::move(scope)};
}

}  // namespace p10::log
