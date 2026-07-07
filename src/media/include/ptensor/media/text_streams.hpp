#pragma once

#include <optional>
#include <vector>

#include <ptensor/p10_result.hpp>

#include "text.hpp"
#include "time/time.hpp"

namespace p10::media {

/// A snapshot of the text (subtitle) streams of a media source.
///
/// Holds every entry of every text stream, grouped per stream. Obtained from
/// MediaCapture::get_text_streams(); building it scans the source once, so it is
/// an immutable copy rather than a live view. Text entries are sparse timed
/// events (heartrate readings, serialized bounding boxes, misc metadata) rather
/// than a per-frame value, which is why the whole set is returned together.
class TextStreams {
  public:
    TextStreams() = default;

    /// Wrap already-collected entries, grouped per stream.
    explicit TextStreams(std::vector<std::vector<Text>> streams) : streams_(std::move(streams)) {}

    /// Number of text streams.
    size_t count() const {
        return streams_.size();
    }

    /// Read all entries of a text stream.
    P10Result<std::vector<Text>> get_text(size_t stream_index) const;

    /// Find the entry active at `timestamp` in a text stream.
    ///
    /// An entry is active over the half-open interval [begin, end). Returns the
    /// first match, or std::nullopt when no entry covers the timestamp. Handy for
    /// looking up per-frame metadata using the frame's own time.
    P10Result<std::optional<Text>> find_text_at(size_t stream_index, const Time& timestamp) const;

  private:
    std::vector<std::vector<Text>> streams_;
};

}  // namespace p10::media
