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

    FfmpegAudioDecoder(AVCodecContext* codec_ctx, int stream_index) :
        codec_ctx_(codec_ctx),
        index_(stream_index) {}

    FfmpegAudioDecoder(const FfmpegAudioDecoder&) = delete;
    FfmpegAudioDecoder& operator=(const FfmpegAudioDecoder&) = delete;

    FfmpegAudioDecoder(FfmpegAudioDecoder&& other) noexcept :
        codec_ctx_(std::exchange(other.codec_ctx_, nullptr)),
        index_(std::exchange(other.index_, -1)) {}

    FfmpegAudioDecoder& operator=(FfmpegAudioDecoder&& other) noexcept {
        if (this != &other) {
            if (codec_ctx_ != nullptr) {
                avcodec_free_context(&codec_ctx_);
            }
            codec_ctx_ = std::exchange(other.codec_ctx_, nullptr);
            index_ = std::exchange(other.index_, -1);
        }
        return *this;
    }

    ~FfmpegAudioDecoder() {
        if (codec_ctx_ != nullptr) {
            avcodec_free_context(&codec_ctx_);
            codec_ctx_ = nullptr;
        }
    }

    static P10Error decode_packet(const AVPacket*, AudioFrame&) {
        return P10Error::NotImplemented;
    }

    int index() const {
        return index_;
    }

  private:
    AVCodecContext* codec_ctx_ = nullptr;
    int index_ = -1;
};

}  // namespace p10::media