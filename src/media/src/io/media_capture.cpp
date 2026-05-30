#include "io/media_capture.hpp"

#include <string>

#include "ffmpeg/ffmpeg_file_media_capture.hpp"

namespace p10::media {

#if defined(__APPLE__)
static constexpr const char* kDeviceFormat = "avfoundation";
#elif defined(__linux__)
static constexpr const char* kDeviceFormat = "v4l2";
#elif defined(_WIN32)
static constexpr const char* kDeviceFormat = "dshow";
#else
static constexpr const char* kDeviceFormat = nullptr;
#endif

static std::string build_device_url(int video_index, int audio_index) {
#if defined(__APPLE__)
    // avfoundation URL: "<video>:<audio>", empty string for absent device
    std::string video = video_index >= 0 ? std::to_string(video_index) : "";
    std::string audio = audio_index >= 0 ? std::to_string(audio_index) : "";
    return video + ":" + audio;
#elif defined(_WIN32)
    // dshow uses named devices; index-based is not standard — callers should
    // use open_file with a dshow URL for Windows.
    return "video=" + std::to_string(video_index);
#else
    // v4l2: /dev/video<n> for video; audio handled separately
    if (video_index >= 0) {
        return "/dev/video" + std::to_string(video_index);
    }
    return "/dev/video0";
#endif
}

void MediaCapture::close() {
    if (impl_) {
        impl_->close();
    }
}

P10Result<MediaCapture> MediaCapture::open_stream(int audio_device_index, int video_device_index) {
    if (kDeviceFormat == nullptr) {
        return Err(P10Error::NotImplemented << "Device capture not supported on this platform");
    }
    if (audio_device_index == NO_DEVICE_SELECTED && video_device_index == NO_DEVICE_SELECTED) {
        return Err(P10Error::InvalidArgument << "At least one device index must be specified");
    }
    const std::string url = build_device_url(video_device_index, audio_device_index);
    auto result = FfmpegFileMediaCapture::open_device(url, kDeviceFormat);
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

P10Result<bool> MediaCapture::next_frame() {
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

std::optional<int64_t> MediaCapture::video_frame_count() const {
    return impl_->video_frame_count();
}

std::optional<double> MediaCapture::duration() const {
    return impl_->duration();
}
}  // namespace p10::media
