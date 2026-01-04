#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

#include "audio_frame.hpp"

namespace p10::media {
class FfmpegAudioDecoder {
  public:
    FfmpegAudioDecoder() = default;

    FfmpegAudioDecoder(AVStream* stream, AVCodecContext* codec_ctx, int stream_index) :
        stream_(stream),
        codec_ctx_(codec_ctx),
        index_(stream_index) {}

    ~FfmpegAudioDecoder() {
        if (codec_ctx_ != nullptr) {
            avcodec_free_context(&codec_ctx_);
            codec_ctx_ = nullptr;
        }
    }

    P10Error decode_packet(const AVPacket*, AudioFrame&) {
        return P10Error::NotImplemented;
    }

    int index() const {
        return index_;
    }

  private:
    AVStream* stream_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    int index_ = -1;
};

}  // namespace p10::media