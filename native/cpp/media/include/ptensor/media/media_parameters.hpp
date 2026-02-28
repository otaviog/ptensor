#pragma once
#include "audio_parameters.hpp"
#include "video_parameters.hpp"

namespace p10::media {
class MediaParameters {
  public:
    const AudioParameters& audio_parameters() const {
        return audio_;
    }

    const VideoParameters& video_parameters() const {
        return video_;
    }

    AudioParameters& audio_parameters() {
        return audio_;
    }

    VideoParameters& video_parameters() {
        return video_;
    }

    MediaParameters& audio_parameters(const AudioParameters& audio) {
        audio_ = audio;
        return *this;
    }

    MediaParameters& video_parameters(const VideoParameters& video) {
        video_ = video;
        return *this;
    }

  private:
    AudioParameters audio_;
    VideoParameters video_;
};
}  // namespace p10::media