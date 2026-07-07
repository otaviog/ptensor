#pragma once

#include <filesystem>
#include <string>

namespace p10::testing {

inline std::filesystem::path get_output_path(const std::string& subdir) {
    std::filesystem::path const output_path = std::filesystem::path("tests/output") / subdir;
    std::filesystem::create_directories(output_path);
    return output_path;
}

inline std::string suffixed(const std::string& filename, const std::string& suffix) {
    std::filesystem::path const path(filename);
    std::string const stem = path.stem().string();
    return stem + "-" + suffix + path.extension().string();
}

}  // namespace p10::testing
