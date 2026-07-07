#pragma once

#include <mutex>
#include <string>
#include <vector>

#include <ptensor/p10_result.hpp>

#include "text.hpp"
#include "text_parameters.hpp"
#include "text_streams.hpp"

extern "C" {
#include <libavformat/avformat.h>
}

namespace p10::media {

/// Discovers and lazily decodes the subtitle/text streams of a capture source.
///
/// Kept out of the audio/video decode engine on purpose: text cues are pulled
/// on demand from a private, throwaway demuxer rather than the live decode
/// thread, so their bookkeeping (source url, stream indices, one-shot scan
/// cache) does not clutter the engine.
class FfmpegTextDecoder {
  public:
    /// Register a re-openable source and the subtitle stream indices found at
    /// open. `url` must be a seekable file (live devices carry no text and
    /// should not call this). Scanning is deferred to the first
    /// get_text_streams() so the open path stays cheap.
    void set_source(std::string url, std::vector<int> stream_indices);

    /// Describe the registered text streams from an already-open demuxer. Cheap:
    /// reads only codec id and language metadata, never triggers a scan.
    std::vector<TextParameters> describe(const AVFormatContext* format_ctx) const;

    /// Snapshot every cue of every registered stream, demuxing the source once
    /// on the first call. Empty when there is no text stream / no source.
    P10Result<TextStreams> get_text_streams() const;

  private:
    /// Demux the source once (on a private throwaway demuxer, so the caller's
    /// live decode thread is untouched) to collect subtitle cues into streams_.
    /// Runs at most once, guarded by mutex_.
    P10Error ensure_scanned() const;

    // Re-openable source and the subtitle stream indices discovered at open.
    // Empty for live device captures.
    std::string source_url_;
    std::vector<int> stream_indices_;

    // Subtitle cues, grouped per text stream. Filled lazily by the first
    // get_text_streams() call; guarded by mutex_ (mutable for the const path).
    mutable std::mutex mutex_;
    mutable bool scanned_ = false;
    mutable std::vector<std::vector<Text>> streams_;
};

}  // namespace p10::media
