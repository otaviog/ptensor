#pragma once

#include <format>
#include <string>
#include <utility>

#include <ptensor/p10_result.hpp>

#include <string_view>

namespace p10::log::detail {

void log_info(std::string_view msg);
void log_warn(std::string_view msg);
void log_error(std::string_view msg);
void log_debug(std::string_view msg);

}  // namespace p10::log::detail

namespace p10::log {

struct ScopedLogger {
    explicit ScopedLogger(std::string scope) noexcept : scope(std::move(scope)) {}

    template<typename... Args>
    void info(std::format_string<Args...> fmt, Args&&... args) const {
        detail::log_info(scope + ": " + std::format(fmt, std::forward<Args>(args)...));
    }

    template<typename... Args>
    void warn(std::format_string<Args...> fmt, Args&&... args) const {
        detail::log_warn(scope + ": " + std::format(fmt, std::forward<Args>(args)...));
    }

    template<typename... Args>
    void error(std::format_string<Args...> fmt, Args&&... args) const {
        detail::log_error(scope + ": " + std::format(fmt, std::forward<Args>(args)...));
    }

    void error(const std::exception& exception) const {
        error("Exception thrown: {}", exception.what());
    }

    void error(const P10Error& err) const {
        error("{}", err.to_string());
    }

    template<typename T>
    void error(const P10Result<T>& res) const {
        if (res.is_error()) {
            error(res.error());
        }
    }

    template<typename... Args>
    void debug(std::format_string<Args...> fmt, Args&&... args) const {
        detail::log_debug(scope + ": " + std::format(fmt, std::forward<Args>(args)...));
    }


    std::string scope;
};

inline ScopedLogger scope(std::string scope) {
    return ScopedLogger {std::move(scope)};
}

}  // namespace p10::log
