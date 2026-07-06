#pragma once
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <ptensor/p10_result.hpp>

#include "../media_parameters.hpp"
#include "../text_streams.hpp"
#include "camera_control.hpp"
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

    // -- Camera property controls -------------------------------------------
    // Only supported by live device captures (see is_stream()); file captures
    // and platforms without a control backend return NotImplemented.

    /// Enable (true) or disable (false) automatic focus.
    P10Error set_auto_focus(bool enabled);
    /// Check whether automatic focus is enabled.
    P10Result<bool> get_auto_focus() const;
    /// Set the manual focus distance. Only takes effect with auto focus off.
    P10Error set_focus_distance(int value);
    /// Get the current focus distance.
    P10Result<int> get_focus_distance() const;
    /// Get the supported range for the focus distance.
    P10Result<CameraControlRange> get_focus_distance_range() const;

    /// Enable (true) or disable (false) automatic exposure.
    P10Error set_auto_exposure(bool enabled);
    /// Check whether automatic exposure is enabled.
    P10Result<bool> get_auto_exposure() const;
    /// Set the manual exposure value. Only takes effect with auto exposure off.
    P10Error set_exposure(int value);
    /// Get the current exposure value.
    P10Result<int> get_exposure() const;
    /// Get the supported range for exposure.
    P10Result<CameraControlRange> get_exposure_range() const;

    /// Set brightness.
    P10Error set_brightness(int value);
    /// Get current brightness.
    P10Result<int> get_brightness() const;
    /// Get the supported range for brightness.
    P10Result<CameraControlRange> get_brightness_range() const;

    /// Set contrast.
    P10Error set_contrast(int value);
    /// Get current contrast.
    P10Result<int> get_contrast() const;
    /// Get the supported range for contrast.
    P10Result<CameraControlRange> get_contrast_range() const;

    /// Set saturation.
    P10Error set_saturation(int value);
    /// Get current saturation.
    P10Result<int> get_saturation() const;
    /// Get the supported range for saturation.
    P10Result<CameraControlRange> get_saturation_range() const;

    /// Set sensor gain.
    P10Error set_gain(int value);
    /// Get current sensor gain.
    P10Result<int> get_gain() const;
    /// Get the supported range for gain.
    P10Result<CameraControlRange> get_gain_range() const;

    /// Enable (true) or disable (false) automatic white balance.
    P10Error set_auto_white_balance(bool enabled);
    /// Check whether automatic white balance is enabled.
    P10Result<bool> get_auto_white_balance() const;
    /// Set the manual white balance color temperature (Kelvin). Only takes
    /// effect with auto white balance off.
    P10Error set_white_balance_temperature(int value);
    /// Get the current white balance color temperature.
    P10Result<int> get_white_balance_temperature() const;
    /// Get the supported range for white balance color temperature.
    P10Result<CameraControlRange> get_white_balance_temperature_range() const;

    /// Set zoom.
    P10Error set_zoom(int value);
    /// Get current zoom.
    P10Result<int> get_zoom() const;
    /// Get the supported range for zoom.
    P10Result<CameraControlRange> get_zoom_range() const;

  private:
    explicit MediaCapture(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

    std::shared_ptr<Impl> impl_;
};
}  // namespace p10::media
