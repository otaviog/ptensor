#include <ptensor/media/text_streams.hpp>

namespace p10::media {

P10Result<std::vector<Text>> TextStreams::get_text(size_t stream_index) const {
    if (stream_index >= streams_.size()) {
        return Err(P10Error::InvalidArgument << "No text stream at the given index");
    }
    return Ok(streams_[stream_index]);
}

P10Result<std::optional<Text>>
TextStreams::find_text_at(size_t stream_index, const Time& timestamp) const {
    if (stream_index >= streams_.size()) {
        return Err(P10Error::InvalidArgument << "No text stream at the given index");
    }

    // Entries are stored in stream (begin) order; a linear scan is fine for the
    // sparse per-frame metadata this targets. An entry covers [begin, end).
    const double seconds = timestamp.to_seconds();
    for (const Text& entry : streams_[stream_index]) {
        if (entry.begin().to_seconds() <= seconds && seconds < entry.end().to_seconds()) {
            return Ok(std::optional<Text>(entry));
        }
    }
    return Ok(std::optional<Text>());
}

}  // namespace p10::media
