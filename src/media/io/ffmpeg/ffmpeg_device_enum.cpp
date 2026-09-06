#include <string>
#include <vector>

#include <ptensor/media/io/media_device.hpp>

#include "../logging.hpp"
#include "ffmpeg_init.hpp"
#include "ffmpeg_wrap_error.hpp"

extern "C" {
#include <libavdevice/avdevice.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
}

namespace p10::media {

#ifdef __APPLE__
// Defined in camera_controls_avf.mm (Objective-C++, AVFoundation). FFmpeg's
// avfoundation demuxer does not implement avdevice_list_input_sources(), so
// video enumeration goes through AVFoundation directly.
P10Result<std::vector<VideoDeviceInfo>> list_avf_video_devices();
#endif

namespace {

#ifdef __APPLE__
    constexpr const char* DEVICE_INPUT_FORMAT = "avfoundation";
#elif defined(__linux__)
    constexpr const char* DEVICE_INPUT_FORMAT = "v4l2";
#elif defined(_WIN32)
    constexpr const char* DEVICE_INPUT_FORMAT = "dshow";
#else
    constexpr const char* DEVICE_INPUT_FORMAT = nullptr;
#endif

    struct RawDeviceEntry {
        std::string name;
        std::string url;
        bool has_video = false;
        bool has_audio = false;
    };

    RawDeviceEntry parse_device_info(const AVDeviceInfo& dev);
    P10Result<const AVInputFormat*> get_input_format();
    P10Result<std::vector<RawDeviceEntry>> enumerate_raw();

}  // namespace

P10Result<std::vector<VideoDeviceInfo>> list_video_devices() {
#ifdef __APPLE__
    return list_avf_video_devices();
#else
    auto raw = enumerate_raw();
    if (raw.is_error()) {
        return Err(raw.error());
    }

    std::vector<VideoDeviceInfo> result;
    int idx = 0;
    for (const auto& entry : raw.unwrap()) {
        if (!entry.has_video) {
            continue;
        }
        VideoDeviceInfo info;
        info.index(idx++).name(entry.name).url(entry.url);
        result.push_back(std::move(info));
    }
    return Ok(std::move(result));
#endif
}

P10Result<std::vector<AudioDeviceInfo>> list_audio_devices() {
    auto raw = enumerate_raw();
    if (raw.is_error()) {
        return Err(raw.error());
    }

    std::vector<AudioDeviceInfo> result;
    int idx = 0;
    for (const auto& entry : raw.unwrap()) {
        if (!entry.has_audio) {
            continue;
        }
        AudioDeviceInfo info;
        info.index(idx++).name(entry.name).url(entry.url);
        result.push_back(std::move(info));
    }
    return Ok(std::move(result));
}

namespace {
    RawDeviceEntry parse_device_info(const AVDeviceInfo& dev) {
        RawDeviceEntry entry;
        if (dev.device_description != nullptr) {
            entry.name = dev.device_description;
        } else if (dev.device_name != nullptr) {
            entry.name = dev.device_name;
        }
        entry.url = dev.device_name != nullptr ? dev.device_name : "";

        for (int j = 0; j < dev.nb_media_types; ++j) {
            if (dev.media_types[j] == AVMEDIA_TYPE_VIDEO) {
                entry.has_video = true;
            } else if (dev.media_types[j] == AVMEDIA_TYPE_AUDIO) {
                entry.has_audio = true;
            }
        }
        // When the backend does not report media types, treat as video by default.
        if (!entry.has_video && !entry.has_audio) {
            entry.has_video = true;
        }

        return entry;
    }

    P10Result<const AVInputFormat*> get_input_format() {
        avdevice_register_all();
        const AVInputFormat* fmt = av_find_input_format(DEVICE_INPUT_FORMAT);
        if (fmt == nullptr) {
            return Err(
                P10Error::NotImplemented
                << (std::string("Unknown input format: ") + DEVICE_INPUT_FORMAT)
            );
        }
        return Ok(fmt);
    }

    P10Result<std::vector<RawDeviceEntry>> enumerate_raw() {
        ffmpeg_init();

        if (DEVICE_INPUT_FORMAT == nullptr) {
            return Err(
                P10Error::NotImplemented << "Device enumeration not supported on this platform"
            );
        }

        auto fmt_result = get_input_format();
        if (fmt_result.is_error()) {
            return Err(fmt_result.error());
        }

        AVDeviceInfoList* list = nullptr;
        const int count = avdevice_list_input_sources(fmt_result.unwrap(), nullptr, nullptr, &list);
        if (count < 0) {
            // Some backends (notably avfoundation) do not implement programmatic
            // source listing and return ENOSYS. Surface an empty list rather than
            // an error so callers can still attempt to open known indices.
            if (count == AVERROR(ENOSYS)) {
                LOGGER.warn("Device enumeration not supported by backend: {}", DEVICE_INPUT_FORMAT);
                if (list != nullptr) {
                    avdevice_free_list_devices(&list);
                }
                return Ok(std::vector<RawDeviceEntry> {});
            }
            if (list != nullptr) {
                avdevice_free_list_devices(&list);
            }
            return Err(wrap_ffmpeg_error(count, "Failed to enumerate input devices"));
        }

        std::vector<RawDeviceEntry> entries;
        if (list != nullptr) {
            entries.reserve(static_cast<size_t>(list->nb_devices));
            for (int i = 0; i < list->nb_devices; ++i) {
                if (list->devices[i] == nullptr) {
                    continue;
                }
                entries.push_back(parse_device_info(*list->devices[i]));
            }
            avdevice_free_list_devices(&list);
        }

        return Ok(std::move(entries));
    }
}  // namespace

}  // namespace p10::media
