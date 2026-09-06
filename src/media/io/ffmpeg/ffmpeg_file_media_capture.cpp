#include "ffmpeg_file_media_capture.hpp"

#include <memory>

#include "ffmpeg_video_decoder.hpp"

extern "C" {
#include <libavformat/avformat.h>
}

namespace p10::media {

P10Result<std::shared_ptr<FfmpegFileMediaCapture>>
FfmpegFileMediaCapture::open(const std::string& path) {
    auto open_result = open_format(path, nullptr, nullptr);
    if (open_result.is_error()) {
        return Err(open_result.error());
    }
    OpenResult const opened = open_result.unwrap();

    auto capture = std::shared_ptr<FfmpegFileMediaCapture>(
        new FfmpegFileMediaCapture(opened.format_ctx, opened.audio_decoder, opened.video_decoder)
    );

    capture->set_text_source(path, opened.text_stream_indices);
    capture->start_decoding_thread();
    return Ok(std::move(capture));
}

std::optional<int64_t> FfmpegFileMediaCapture::video_frame_count() const {
    if (video_decoder() != nullptr) {
        return video_decoder()->video_frame_count();
    }
    return std::nullopt;
}

std::optional<double> FfmpegFileMediaCapture::duration() const {
    if (format_ctx() != nullptr) {
        if (format_ctx()->duration != AV_NOPTS_VALUE) {
            return format_ctx()->duration / static_cast<double>(AV_TIME_BASE);
        }
    }
    return std::nullopt;
}

P10Error FfmpegFileMediaCapture::seek(double seconds) {
    return seek_to(seconds);
}

}  // namespace p10::media
