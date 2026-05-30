#include "model_downloader.hpp"

#include <filesystem>
#include <fstream>

#include <curl/curl.h>

#include "ptensor/p10_error.hpp"
#include <ptensor/log/log.hpp>

namespace p10::infer {

namespace {

    std::string get_user_app_dir() {
#if defined(_WIN32)
        const char* appdata = std::getenv("APPDATA");
        if (appdata) {
            return std::string(appdata);
        }
        return ".";
#elif defined(__APPLE__)
        const char* home = std::getenv("HOME");
        if (home) {
            return std::string(home) + "/Library/Application Support";
        }
        return ".";
#else
        // XDG_DATA_HOME or ~/.local/share
        const char* xdg = std::getenv("XDG_DATA_HOME");
        if (xdg && xdg[0] != '\0') {
            return std::string(xdg);
        }
        const char* home = std::getenv("HOME");
        if (home) {
            return std::string(home) + "/.local/share";
        }
        return ".";
#endif
    }

    size_t write_callback(void* ptr, size_t size, size_t nmemb, void* userdata) {
        auto* file = static_cast<std::ofstream*>(userdata);
        const size_t bytes = size * nmemb;
        file->write(static_cast<const char*>(ptr), static_cast<std::streamsize>(bytes));
        return bytes;
    }

    P10Result<std::string>
    download_file(const std::string& url, const std::filesystem::path& dest_path) {
        std::ofstream file(dest_path, std::ios::binary);
        if (!file.is_open()) {
            return Err(
                P10Error::IoError << ("Cannot open file for writing: " + dest_path.string())
            );
        }

        CURL* curl = curl_easy_init();
        if (!curl) {
            return Err(P10Error::UnknownError << "Failed to initialize libcurl");
        }

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &file);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);

        const CURLcode res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        file.close();

        if (res != CURLE_OK) {
            std::filesystem::remove(dest_path);
            return Err(
                P10Error::IoError << ("Download failed for " + url + ": " + curl_easy_strerror(res))
            );
        }

        return Ok(dest_path.string());
    }

}  // namespace

bool is_url(const std::string& path) {
    return path.rfind("http://", 0) == 0 || path.rfind("https://", 0) == 0;
}

P10Result<std::string> resolve_model_path(const std::string& path) {
    if (!is_url(path)) {
        return Ok(std::string(path));
    }

    // Extract filename from URL (last path segment, strip query string).
    std::string url_path = path;
    const auto query_pos = url_path.find('?');
    if (query_pos != std::string::npos) {
        url_path = url_path.substr(0, query_pos);
    }
    const auto slash_pos = url_path.rfind('/');
    const std::string filename =
        (slash_pos != std::string::npos) ? url_path.substr(slash_pos + 1) : "model.onnx";

    const std::filesystem::path models_dir =
        std::filesystem::path(get_user_app_dir()) / "ptensor" / "models";

    std::error_code ec;
    std::filesystem::create_directories(models_dir, ec);
    if (ec) {
        return Err(P10Error::IoError << ("Cannot create models directory: " + ec.message()));
    }

    const std::filesystem::path dest = models_dir / filename;

    if (std::filesystem::exists(dest)) {
        log::scope("model_downloader").info("Model already cached: {}", dest.string());
        return Ok(dest.string());  // string() returns a new std::string (rvalue)
    }

    log::scope("model_downloader").info("Downloading model from {} to {}", path, dest.string());
    return download_file(path, dest);
}

}  // namespace p10::infer
