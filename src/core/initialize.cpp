#include "initialize.hpp"

namespace p10 {
namespace {
    std::string g_log_directory = "./ptensor-logs";
}

std::string get_log_directory() {
    return g_log_directory;
}

void initialize(const std::string& log_directory) {
    g_log_directory = log_directory;
}
}  // namespace p10
