#pragma once
#include <string>
#include <vector>

#include <ptensor/p10_result.hpp>

#include "../audio_parameters.hpp"
#include "../time/rational.hpp"
#include "../video_parameters.hpp"

namespace p10::media {

namespace detail {
    template<typename BaseClass>
    class DeviceInfo {
      public:
        int index() const {
            return index_;
        }

        /// Human-readable device name (e.g. "FaceTime HD Camera").
        const std::string& name() const {
            return name_;
        }

        /// Backend-specific identifier passed to FFmpeg when opening.
        const std::string& url() const {
            return url_;
        }

        BaseClass& index(int index) {
            index_ = index;
            return static_cast<BaseClass&>(*this);
        }

        BaseClass& name(std::string name) {
            name_ = std::move(name);
            return static_cast<BaseClass&>(*this);
        }

        BaseClass& url(std::string url) {
            url_ = std::move(url);
            return static_cast<BaseClass&>(*this);
        }

      private:
        DeviceInfo() = default;
        friend BaseClass;

        int index_ = -1;
        std::string name_;
        std::string url_;
    };
}  // namespace detail

class VideoCapability {
  public:
    int width() const {
        return width_;
    }

    int height() const {
        return height_;
    }

    Rational min_frame_rate() const {
        return min_frame_rate_;
    }

    Rational max_frame_rate() const {
        return max_frame_rate_;
    }

    VideoCapability& width(int width) {
        width_ = width;
        return *this;
    }

    VideoCapability& height(int height) {
        height_ = height;
        return *this;
    }

    VideoCapability& min_frame_rate(const Rational& frame_rate) {
        min_frame_rate_ = frame_rate;
        return *this;
    }

    VideoCapability& max_frame_rate(const Rational& frame_rate) {
        max_frame_rate_ = frame_rate;
        return *this;
    }

  private:
    size_t width_;
    size_t height_;
    Rational min_frame_rate_;
    Rational max_frame_rate_;
    // TODO: include format here?
};

class VideoDeviceInfo: public detail::DeviceInfo<VideoDeviceInfo> {
  public:
    const std::vector<VideoCapability>& capabilities() const {
        return capabilities_;
    }

    VideoDeviceInfo& add_capability(const VideoCapability& capability) {
        capabilities_.push_back(capability);
        return *this;
    }

    VideoParameters match_closest(int width, int height, Rational frame_rate) const;

  private:
    std::vector<VideoCapability> capabilities_;
};

class AudioDeviceInfo: public detail::DeviceInfo<AudioDeviceInfo> {
  public:
    const std::vector<AudioParameters>& capabilities() const {
        return capabilities_;
    }

    AudioDeviceInfo& add_capability(const AudioParameters& capability) {
        capabilities_.push_back(capability);
        return *this;
    }

    static constexpr double SAMPLE_RATE_ANY = 0.0;
    static constexpr size_t NUMBER_OF_CHANNELS_ANY = 0;
    /// Find the closest audio configuration to that device supports.
    ///    If both parameters are ANY, then return the default.
    ///
    /// # Arguments
    ///
    /// * sample_rate - target sample rate, pass SAMPLE_RATE_ANY to consider any.
    /// * num_channels - number of channels, pass NUMBER_CHANNELS_ANY if dont care
    ///
    /// # Returns
    ///
    /// * The closest audio capability. Returns the default audio parameters if it's empty,
    ///   note the list_audio_devices never return devices with no capabilities.
    AudioParameters match_closest(double sample_rate, size_t num_channels);

  private:
    std::vector<AudioParameters> capabilities_;
};

/// Enumerate the capture devices available through the platform's FFmpeg input
/// device backend (avfoundation on macOS, v4l2 on Linux, dshow on Windows).
///
/// Capability lists are populated on a best-effort basis: some backends report
/// rich capability data, others report only the device list. An empty
/// capability list means "unknown", not "none".
P10Result<std::vector<VideoDeviceInfo>> list_video_devices();

P10Result<std::vector<AudioDeviceInfo>> list_audio_devices();

}  // namespace p10::media
