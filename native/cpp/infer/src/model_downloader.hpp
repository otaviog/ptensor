#pragma once

#include <string>

#include <ptensor/p10_result.hpp>

namespace p10::infer {

// Returns true if the given path looks like an HTTP(S) URL.
bool is_url(const std::string& path);

// If path is a URL, downloads the model to USER_APP_DIR/ptensor/models/<filename>
// (skips the download if the file already exists) and returns the local path.
// If path is not a URL, returns it unchanged.
P10Result<std::string> resolve_model_path(const std::string& path);

}  // namespace p10::infer
