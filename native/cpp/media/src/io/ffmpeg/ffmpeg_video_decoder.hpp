#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

#include "ffmpeg_sws.hpp"
#include "video_frame.hpp"
#include "video_parameters.hpp"

namespace p10::media {
class VideoQueue;

class FfmpegVideoDecoder {
  public:
    enum class DecodeStatus { FrameDecoded = 0, Eof, Again, Cancelled };
    FfmpegVideoDecoder() = default;

    FfmpegVideoDecoder(AVStream* stream, AVCodecContext* codec_ctx, int stream_index) :
        stream_(stream),
        codec_ctx_(codec_ctx),
        index_(stream_index) {}

    ~FfmpegVideoDecoder() {
        avcodec_free_context(&codec_ctx_);
        codec_ctx_ = nullptr;
        stream_ = nullptr;
    }

    P10Result<DecodeStatus> decode_packet(const AVPacket* pkt, VideoQueue& queue);

    int index() const {
        return index_;
    }

    VideoParameters get_video_parameters() const;

  private:
    AVStream* stream_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    int index_ = -1;
    FfmpegSws sws_converter;
};

}  // namespace p10::media