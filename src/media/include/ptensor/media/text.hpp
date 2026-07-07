#pragma once

#include <string>
#include <utility>

#include "time/time.hpp"

namespace p10::media {
/// A single timed text entry: a string payload shown between two timestamps.
///
/// Used to carry overlay-style metadata alongside video/audio (heartrate
/// readings, serialized bounding boxes, misc tags). It maps to one subtitle
/// cue when written to / read from a media container.
class Text {
  public:
    /// Create a text entry with an optional active interval.
    Text(std::string text, Time begin = Time(), Time end = Time()) :
        text_(std::move(text)),
        begin_(begin),
        end_(end) {}

    /// Get the text payload.
    const std::string& text() const {
        return text_;
    }

    /// Get the time the entry becomes active.
    Time begin() const {
        return begin_;
    }

    /// Get the time the entry stops being active.
    Time end() const {
        return end_;
    }

  private:
    std::string text_;
    Time begin_;
    Time end_;
};
}  // namespace p10::media
