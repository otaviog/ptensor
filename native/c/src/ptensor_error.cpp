

#include <string>

#include <ptensor/ptensor_error.hpp>

#include "update_error_state.hpp"

namespace {
thread_local std::string g_error_message;
}  // namespace

namespace p10 {
P10ErrorEnum update_error_state(
    const PtensorError& error) {
    if (error.is_ok()) {
        return P10ErrorEnum::P10_OK;
    }

    g_error_message = error.to_string();
    return static_cast<P10ErrorEnum>(error.code());
}
}

PTENSOR_API const char* p10_get_last_error_message() {
    if (g_error_message.empty()) {
        return nullptr;
    }
    return g_error_message.c_str();
}