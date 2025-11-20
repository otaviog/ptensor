#include "io/media_capture.hpp"

#include "ffmpeg/ffmpeg_file_media_capture.hpp"
#include "ptensor/p10_result.hpp"

namespace p10::media {

P10Result<MediaCapture> MediaCapture::open_file(const std::string& path) {
    auto result = FfmpegFileMediaCapture::open(path);
    if (result.is_error()) {
        return Err(result.err());
    }
    return Ok(MediaCapture(result.unwrap()));
}
}  // namespace p10::media