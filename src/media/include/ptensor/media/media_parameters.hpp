#pragma once
#include "audio_parameters.hpp"
#include "video_parameters.hpp"

namespace p10::media {
/// Combined audio and video parameters.
class MediaParameters {
  public:
    /// Get audio parameters.
    const AudioParameters& audio_parameters() const {
        return audio_;
    }

    /// Get video parameters.
    const VideoParameters& video_parameters() const {
        return video_;
    }

    /// Get mutable audio parameters.
    AudioParameters& audio_parameters() {
        return audio_;
    }

    /// Get mutable video parameters.
    VideoParameters& video_parameters() {
        return video_;
    }

    /// Set audio parameters.
    MediaParameters& audio_parameters(const AudioParameters& audio) {
        audio_ = audio;
        return *this;
    }

    /// Set video parameters.
    MediaParameters& video_parameters(const VideoParameters& video) {
        video_ = video;
        return *this;
    }

  private:
    AudioParameters audio_;
    VideoParameters video_;
};
}  // namespace p10::media