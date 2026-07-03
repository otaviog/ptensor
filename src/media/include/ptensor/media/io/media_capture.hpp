#pragma once
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <ptensor/p10_result.hpp>

#include "../media_parameters.hpp"
#include "../text_streams.hpp"
#include "media_device.hpp"

namespace p10::media {
class VideoFrame;
class AudioFrame;

/// Media capture from files or live devices.
class MediaCapture {
  public:
    /// Result of next_frame() operation.
    enum NextFrameResult {
        Available,  ///< Frame decoded and ready; call get_video()/get_audio().
        NotReady,  ///< No frame queued yet; only returned in Poll mode.
        Done  ///< End of file or stream stopped; no more frames.
    };

    /// Waiting behavior for next_frame().
    enum WaitMode {
        Poll,  ///< Non-blocking: returns NotReady immediately if queue is empty.
        Block  ///< Blocking: waits until a frame is available; never returns NotReady.
    };

    class Impl;

    // MediaCapture instances are created by open_file() or open_stream().

    /// Open a media file for reading.
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

    /// Close the media capture.
    void close();

    /// Get current media parameters.
    MediaParameters get_parameters() const;

    /// Check if this is a live stream (not a file).
    bool is_stream() const;

    /// Retrieve the next frame.
    P10Result<NextFrameResult> next_frame(WaitMode wait = WaitMode::Poll);

    /// Get the most recently decoded video frame.
    P10Error get_video(VideoFrame& frame);

    /// Get the most recently decoded audio frame.
    P10Error get_audio(AudioFrame& frame);

    /// Read the source's text (subtitle) streams.
    ///
    /// Returns a TextStreams snapshot exposing count(), get_text() and
    /// find_text_at(). For file captures the streams are scanned on this call
    /// (once); live device captures carry no text and yield an empty snapshot.
    P10Result<TextStreams> get_text_streams() const;

    /// Get total video frame count (if known).
    std::optional<int64_t> video_frame_count() const;

    /// Get total duration in seconds (if known).
    std::optional<double> duration() const;

    /// Seek to position in seconds from the start. File captures only.
    P10Error seek(double seconds);

  private:
    explicit MediaCapture(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

    std::shared_ptr<Impl> impl_;
};
}  // namespace p10::media
