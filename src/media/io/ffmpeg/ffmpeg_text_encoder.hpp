#pragma once

#include <ptensor/p10_result.hpp>

#include "ffmpeg_memory.hpp"
#include "time/rational.hpp"

struct AVFormatContext;
struct AVStream;

namespace p10::media {
class TextParameters;
class Text;

/// Muxes timed text into a subtitle stream.
///
/// Text subtitles (SubRip) are stored as raw UTF-8 packets: the cue text is the
/// packet payload and the interval lives in the packet pts/duration. No codec
/// context is involved, so this "encoder" only declares the stream and turns a
/// Text into a stream-timed AVPacket. This is reliable in Matroska; MP4 would
/// need mov_text (a length-prefixed bitstream) instead.
class FfmpegTextEncoder {
  public:
    FfmpegTextEncoder() = default;
    FfmpegTextEncoder(const FfmpegTextEncoder&) = delete;
    FfmpegTextEncoder& operator=(const FfmpegTextEncoder&) = delete;
    FfmpegTextEncoder(FfmpegTextEncoder&&) = delete;
    FfmpegTextEncoder& operator=(FfmpegTextEncoder&&) = delete;
    ~FfmpegTextEncoder() = default;

    /// Add a subtitle stream to `output_format` for the given parameters.
    P10Error create(const TextParameters& text_params, AVFormatContext* output_format);

    /// Build a stream-timed packet for `text` (caller writes it to the muxer).
    P10Result<UniqueAvPacket> encode(const Text& text) const;

    AVStream* stream() const {
        return stream_;
    }

  private:
    AVStream* stream_ = nullptr;
    Rational time_base_ {1, 1000};
};

}  // namespace p10::media
