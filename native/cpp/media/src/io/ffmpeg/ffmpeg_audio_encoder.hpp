#pragma once

#include <ptensor/p10_error.hpp>

#include "audio_parameters.hpp"
#include "ffmpeg_audio_fifo.hpp"
#include "ffmpeg_swr.hpp"

struct AVFormatContext;
struct AVStream;
struct AVCodecContext;

namespace p10::media {
class AudioParameters;

class FfmpegAudioEncoder {
  public:
    P10Error create(const AudioParameters& audio_params, AVStream* audio_stream);

  private:
    AVStream* stream_ = nullptr;
    AVCodecContext* codec_context_ = nullptr;
    FfmpegSwr resampler_;
    FfmpegAudioFifo fifo_;
    int64_t m_audioCurrentPts = 0;
    int64_t m_videoAudioSamplesDiff = 0;
};
}  // namespace p10::media