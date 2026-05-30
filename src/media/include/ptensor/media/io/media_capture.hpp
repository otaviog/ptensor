#pragma once
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <ptensor/p10_result.hpp>

#include "../media_parameters.hpp"
#include "media_device.hpp"

namespace p10::media {
class VideoFrame;
class AudioFrame;

class MediaCapture {
  public:
    enum NextFrameResult {
        Available,  ///< Frame decoded and ready; call get_video()/get_audio().
        NotReady,  ///< No frame queued yet; only returned in Poll mode.
        Done  ///< End of file or stream stopped; no more frames.
    };

    enum WaitMode {
        Poll,  ///< Non-blocking: returns NotReady immediately if queue is empty.
        Block  ///< Blocking: waits until a frame is available; never returns NotReady.
    };

    class Impl;

    MediaCapture() = default;

    static P10Result<MediaCapture> open_file(const std::string& path);

    /// Enumerate the video capture devices available on this platform.
    static P10Result<std::vector<VideoDeviceInfo>> list_video_devices();

    /// Enumerate the audio capture devices available on this platform.
    static P10Result<std::vector<AudioDeviceInfo>> list_audio_devices();

    /// Open a live capture stream. Each optional bundles a device index with
    /// its desired parameters (use VideoDeviceInfo::match_closest to obtain
    /// them). Pass std::nullopt to omit a device type.
    static P10Result<MediaCapture> open_stream(
        std::optional<std::pair<int, VideoParameters>> video = std::nullopt,
        std::optional<std::pair<int, AudioParameters>> audio = std::nullopt
    );

    void close();

    MediaParameters get_parameters() const;

    bool is_stream() const;

    P10Result<NextFrameResult> next_frame(WaitMode wait = WaitMode::Poll);

    P10Error get_video(VideoFrame& frame);

    P10Error get_audio(AudioFrame& frame);

    std::optional<int64_t> video_frame_count() const;

    std::optional<double> duration() const;

    /// Seek to `seconds` from the start of the stream. Only supported for
    /// file-based captures; returns NotImplemented for live device captures.
    P10Error seek(double seconds);

  private:
    explicit MediaCapture(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

    std::shared_ptr<Impl> impl_;
};
}  // namespace p10::media
