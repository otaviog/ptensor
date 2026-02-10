#pragma once

#include <queue>

#include <ptensor/p10_error.hpp>

#include "audio_parameters.hpp"
#include "ffmpeg_audio_fifo.hpp"
#include "ffmpeg_swr.hpp"

struct AVFormatContext;
struct AVStream;
struct AVCodecContext;
struct AVPacket;

namespace p10::media {
class AudioParameters;

class FfmpegAudioEncoder {
  public:
    FfmpegAudioEncoder() = default;
    ~FfmpegAudioEncoder();

    P10Error create(const AudioParameters& audio_params, AVFormatContext* format_ctx);

    void reset();

    P10Error encode(const AudioFrame& frame);

    P10Error flush();

    bool has_packets() const {
        return !encoded_packets_.empty();
    }

    AVPacket* pop_encoded_packet() {
        if (encoded_packets_.empty()) {
            return nullptr;
        }
        AVPacket* pkt = encoded_packets_.front();
        encoded_packets_.pop();
        return pkt;
    }

    AVStream* stream() const {
        return stream_;
    }

    AVCodecContext* codec_context() const {
        return codec_context_;
    }

  private:
    P10Error flush_encoding_fifo();
    P10Error receive_packets();

    AVStream* stream_ = nullptr;
    AVCodecContext* codec_context_ = nullptr;
    FfmpegSwr resampler_;
    FfmpegAudioFifo encoding_fifo_;
    int64_t audio_pts_ = 0;
    std::queue<AVPacket*> encoded_packets_;
};
}  // namespace p10::media
