#pragma once
#include <utility>
#include <vector>

#include "audio_parameters.hpp"
#include "text_parameters.hpp"
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

    /// Get the declared text streams.
    const std::vector<TextParameters>& text_parameters() const {
        return text_;
    }

    /// Get mutable text streams.
    std::vector<TextParameters>& text_parameters() {
        return text_;
    }

    /// Replace the text streams.
    MediaParameters& text_parameters(std::vector<TextParameters> text) {
        text_ = std::move(text);
        return *this;
    }

    /// Append one text stream and return its index.
    MediaParameters& add_text_stream(const TextParameters& text) {
        text_.push_back(text);
        return *this;
    }

  private:
    AudioParameters audio_;
    VideoParameters video_;
    std::vector<TextParameters> text_;
};
}  // namespace p10::media