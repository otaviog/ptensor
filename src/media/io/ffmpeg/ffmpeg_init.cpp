#include "ffmpeg_init.hpp"

#include <cstdio>
#include <filesystem>
#include <mutex>

extern "C" {
#include <libavutil/log.h>
}

#include "ptensor/initialize.hpp"

namespace p10::media {
namespace {
    std::FILE* g_log_file = nullptr;
    std::mutex g_log_mutex;

    void ffmpeg_log_callback(void* avcl, int level, const char* fmt, va_list args);
}  // namespace

void ffmpeg_init() {
    static std::once_flag flag;
    std::call_once(flag, [] {
        std::filesystem::path const log_dir(get_log_directory());
        std::error_code ec;
        std::filesystem::create_directories(log_dir, ec);

        g_log_file = std::fopen((log_dir / "ffmpeg.log").string().c_str(), "a");
        if (g_log_file == nullptr) {
            // Keep FFmpeg's default stderr logging.
            return;
        }

        av_log_set_level(AV_LOG_INFO);
        av_log_set_callback(ffmpeg_log_callback);
    });
}

namespace {
    void ffmpeg_log_callback(void* /*avcl*/, int level, const char* fmt, va_list args) {
        if (level > av_log_get_level()) {
            return;
        }

        std::scoped_lock const guard(g_log_mutex);
        if (g_log_file != nullptr) {
            static_cast<void>(std::vfprintf(g_log_file, fmt, args));
            static_cast<void>(std::fflush(g_log_file));
        }
    }
}  // namespace
}  // namespace p10::media
