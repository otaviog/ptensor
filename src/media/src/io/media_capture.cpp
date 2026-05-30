#include "io/media_capture.hpp"

#include <string>
#include <utility>

#include "ffmpeg/ffmpeg_device_media_capture.hpp"
#include "ffmpeg/ffmpeg_file_media_capture.hpp"

namespace p10::media {

void MediaCapture::close() {
    if (impl_) {
        impl_->close();
    }
}

P10Result<std::vector<VideoDeviceInfo>> MediaCapture::list_video_devices() {
    return p10::media::list_video_devices();
}

P10Result<std::vector<AudioDeviceInfo>> MediaCapture::list_audio_devices() {
    return p10::media::list_audio_devices();
}

P10Result<MediaCapture> MediaCapture::open_stream(
    std::optional<std::pair<int, VideoParameters>> video,
    std::optional<std::pair<int, AudioParameters>> audio
) {
    auto result = FfmpegDeviceMediaCapture::open(std::move(video), std::move(audio));
    if (result.is_error()) {
        return Err(result.error());
    }
    return Ok(MediaCapture(result.unwrap()));
}

P10Result<MediaCapture> MediaCapture::open_file(const std::string& path) {
    auto result = FfmpegFileMediaCapture::open(path);
    if (result.is_error()) {
        return Err(result.error());
    }
    return Ok(MediaCapture(result.unwrap()));
}

P10Result<MediaCapture::NextFrameResult> MediaCapture::next_frame(WaitMode wait) {
    return impl_->next_frame(wait);
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

bool MediaCapture::is_stream() const {
    return dynamic_cast<FfmpegDeviceMediaCapture*>(impl_.get()) != nullptr;
}

std::optional<int64_t> MediaCapture::video_frame_count() const {
    return impl_->video_frame_count();
}

std::optional<double> MediaCapture::duration() const {
    return impl_->duration();
}

P10Error MediaCapture::seek(double seconds) {
    if (!impl_) {
        return P10Error::InvalidArgument << "MediaCapture not open";
    }
    return impl_->seek(seconds);
}

}  // namespace p10::media
