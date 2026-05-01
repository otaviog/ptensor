#pragma once

#include <string>
#include <ng-log/logging.h>
#include <ptensor/p10_result.hpp>

namespace p10::log {

inline P10Error init() {
    return P10Error::Ok;
}

struct ScopedLogger {
    explicit ScopedLogger(std::string scope) : scope(std::move(scope)) {}

    void info(const std::string& message) const {
        LOG(INFO) << scope << ": " << message;
    }

    void error(const std::string& message) const {
        LOG(ERROR) << scope << ": " << message;
    }

    void error(const std::exception& exeception) const {
        LOG(ERROR) << scope << ": Exception throw" << exeception.what();
    }

    std::string scope;
};

inline ScopedLogger scope(std::string scope) {
    return ScopedLogger {std::move(scope)};
}

}  // namespace p10::log
