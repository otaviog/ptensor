

#include <string>

#include <ptensor/p10_error.hpp>

#include "update_error_state.hpp"

namespace {
thread_local std::string g_error_message;
}  // namespace

namespace p10 {
P10ErrorEnum update_error_state(const P10Error& error) {
    if (!error.is_ok()) {
        g_error_message = error.to_string();
    } else {
        g_error_message.clear();
    }

    return static_cast<P10ErrorEnum>(error.code());
}
}  // namespace p10

PTENSOR_API const char* p10_get_last_error_message() {
    if (g_error_message.empty()) {
        return nullptr;
    }
    return g_error_message.c_str();
}
