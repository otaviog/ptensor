#include "io/media_capture.hpp"

#include "ffmpeg/ffmpeg_file_media_capture.hpp"
#include "ptensor/p10_error.hpp"
#include "ptensor/p10_result.hpp"

namespace p10::media {

void MediaCapture::close() {
    impl_->close();
}

P10Result<MediaCapture> MediaCapture::open_file(const std::string& path) {
    auto result = FfmpegFileMediaCapture::open(path);
    if (result.is_error()) {
        return Err(result.error());
    }
    return Ok(MediaCapture(result.unwrap()));
}

P10Error MediaCapture::next_frame() {
    return impl_->next_frame();
}

P10Error MediaCapture::get_video(VideoFrame& frame) {
    return impl_->get_video(frame);
}

P10Error MediaCapture::get_audio(AudioFrame& frame) {
    return impl_->get_audio(frame);
}

MediaParameters MediaCapture::get_parameters() const {
    return impl_->get_parameters();
}
}  // namespace p10::media