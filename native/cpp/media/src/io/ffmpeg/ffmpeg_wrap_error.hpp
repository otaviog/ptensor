#pragma once
#include <array>
extern "C" {
#include <libavutil/error.h>
}
#include <ptensor/p10_error.hpp>

namespace p10::media {
inline P10Error wrap_ffmpeg_error(int ffmpeg_error_code, const std::string& context_message = "") {
    if (ffmpeg_error_code >= 0) {
        return P10Error::Ok;
    }

    std::array<char, AV_ERROR_MAX_STRING_SIZE> error_buffer;
    av_strerror(ffmpeg_error_code, error_buffer.data(), error_buffer.size());

    std::string full_message = context_message;
    if (!context_message.empty()) {
        full_message += ": ";
    }
    full_message += std::string(error_buffer.data());

    return P10Error(P10Error::IoError, full_message);
}

}  // namespace p10::media