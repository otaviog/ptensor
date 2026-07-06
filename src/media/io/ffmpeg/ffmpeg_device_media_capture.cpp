#include "ffmpeg_device_media_capture.hpp"

#include <memory>
#include <string>

extern "C" {
#include <libavdevice/avdevice.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
}

namespace p10::media {

namespace {

#if defined(__APPLE__)
    constexpr const char* DEVICE_INPUT_FORMAT = "avfoundation";
#elif defined(__linux__)
    constexpr const char* DEVICE_INPUT_FORMAT = "v4l2";
#elif defined(_WIN32)
    constexpr const char* DEVICE_INPUT_FORMAT = "dshow";
#else
    constexpr const char* DEVICE_INPUT_FORMAT = nullptr;
#endif

    std::string build_device_url(int video_index, int audio_index);
    void apply_video_options(const VideoParameters& params, AVDictionary** options);
    void apply_audio_options(const AudioParameters& params, AVDictionary** options);

}  // namespace

P10Result<std::shared_ptr<FfmpegDeviceMediaCapture>> FfmpegDeviceMediaCapture::open(
    std::optional<std::pair<int, VideoParameters>> video,
    std::optional<std::pair<int, AudioParameters>> audio
) {
    if (DEVICE_INPUT_FORMAT == nullptr) {
        return Err(P10Error::NotImplemented << "Device capture not supported on this platform");
    }

    const int video_device_index = video ? video->first : -1;
    const int audio_device_index = audio ? audio->first : -1;

    if (video_device_index < 0 && audio_device_index < 0) {
        return Err(P10Error::InvalidArgument << "At least one device index must be specified");
    }

    avdevice_register_all();
    const AVInputFormat* fmt = av_find_input_format(DEVICE_INPUT_FORMAT);
    if (fmt == nullptr) {
        return Err(
            P10Error::NotImplemented
            << (std::string("Unknown input format: ") + DEVICE_INPUT_FORMAT)
        );
    }

    const std::string url = build_device_url(video_device_index, audio_device_index);

    AVDictionary* options = nullptr;
    if (video.has_value()) {
        apply_video_options(video->second, &options);
    } else if (video_device_index >= 0) {
#if defined(__APPLE__)
        av_dict_set(&options, "framerate", "30", 0);
#endif
    }
    if (audio.has_value()) {
        apply_audio_options(audio->second, &options);
    }

    auto open_result = open_format(url, fmt, &options);
    av_dict_free(&options);
    if (open_result.is_error()) {
        return Err(open_result.error());
    }
    OpenResult opened = open_result.unwrap();

    auto capture = std::shared_ptr<FfmpegDeviceMediaCapture>(new FfmpegDeviceMediaCapture(
        opened.format_ctx,
        opened.audio_decoder,
        opened.video_decoder,
        open_camera_control_backend(video_device_index)
    ));
    capture->start_decoding_thread();
    return Ok(std::move(capture));
}

P10Result<int> FfmpegDeviceMediaCapture::get_camera_control(CameraControlId id) const {
    if (!camera_controls_) {
        return Err(
            P10Error::NotImplemented << "Camera controls not supported on this platform/device"
        );
    }
    return camera_controls_->get(id);
}

P10Error FfmpegDeviceMediaCapture::set_camera_control(CameraControlId id, int value) {
    if (!camera_controls_) {
        return P10Error::NotImplemented << "Camera controls not supported on this platform/device";
    }
    return camera_controls_->set(id, value);
}

P10Result<CameraControlRange> FfmpegDeviceMediaCapture::get_camera_control_range(CameraControlId id
) const {
    if (!camera_controls_) {
        return Err(
            P10Error::NotImplemented << "Camera controls not supported on this platform/device"
        );
    }
    return camera_controls_->get_range(id);
}

P10Result<bool> FfmpegDeviceMediaCapture::get_camera_auto_control(CameraAutoControlId id) const {
    if (!camera_controls_) {
        return Err(
            P10Error::NotImplemented << "Camera controls not supported on this platform/device"
        );
    }
    return camera_controls_->get_auto(id);
}

P10Error FfmpegDeviceMediaCapture::set_camera_auto_control(CameraAutoControlId id, bool enabled) {
    if (!camera_controls_) {
        return P10Error::NotImplemented << "Camera controls not supported on this platform/device";
    }
    return camera_controls_->set_auto(id, enabled);
}

namespace {
    std::string build_device_url(int video_index, int audio_index) {
#if defined(__APPLE__)
        // avfoundation URL: "<video>:<audio>", empty string for absent device.
        std::string video = video_index >= 0 ? std::to_string(video_index) : "";
        std::string audio = audio_index >= 0 ? std::to_string(audio_index) : "";
        return video + ":" + audio;
#elif defined(_WIN32)
        (void)audio_index;
        return "video=" + std::to_string(video_index);
#else
        (void)audio_index;
        return video_index >= 0 ? "/dev/video" + std::to_string(video_index) : "/dev/video0";
#endif
    }

    void apply_video_options(const VideoParameters& params, AVDictionary** options) {
        if (params.width() > 0 && params.height() > 0) {
            const std::string size =
                std::to_string(params.width()) + "x" + std::to_string(params.height());
            av_dict_set(options, "video_size", size.c_str(), 0);
        }
        if (params.frame_rate().num() > 0) {
            const std::string rate = std::to_string(params.frame_rate().num()) + "/"
                + std::to_string(params.frame_rate().den());
            av_dict_set(options, "framerate", rate.c_str(), 0);
        } else {
#if defined(__APPLE__)
            // avfoundation defaults to 29.97 which most cameras don't support; 30 matches
            // the ~30fps modes avfoundation typically advertises (e.g. 30.000030fps).
            av_dict_set(options, "framerate", "30", 0);
#endif
        }
        if (!params.pixel_format().empty() && params.pixel_format() != "rgb") {
            av_dict_set(options, "pixel_format", params.pixel_format().c_str(), 0);
        }
    }

    void apply_audio_options(const AudioParameters& params, AVDictionary** options) {
        if (params.audio_sample_rate_hz() > 0.0) {
            av_dict_set(
                options,
                "sample_rate",
                std::to_string(static_cast<int>(params.audio_sample_rate_hz())).c_str(),
                0
            );
        }
        if (params.audio_channels() > 0) {
            av_dict_set(options, "channels", std::to_string(params.audio_channels()).c_str(), 0);
        }
    }
}  // namespace

}  // namespace p10::media
