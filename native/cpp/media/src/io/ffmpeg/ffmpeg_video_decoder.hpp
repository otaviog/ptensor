#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

#include "ffmpeg_memory.hpp"
#include "ffmpeg_sws.hpp"
#include "ffmpeg_wrap_error.hpp"
#include "video_frame.hpp"

namespace p10::media {
class FfmpegVideoDecoder {
  public:
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

    P10Error decode_packet(const AVPacket* pkt, VideoFrame& out_frame) {
        while (true) {
            UniqueAvFrame frame(av_frame_alloc());
            int ret = avcodec_send_packet(codec_ctx_, pkt);
            if (ret < 0) {
                return wrap_error(ret);
            }

            ret = avcodec_receive_frame(codec_ctx_, frame.get());
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break;
            } else if (ret < 0) {
                return wrap_error(ret);
            }

            return sws_converter.transform(frame.get(), out_frame);
        }

        return P10Error::Ok;
    }

    int index() const {
        return index_;
    }

  private:
    AVStream* stream_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    int index_ = -1;
    FfmpegSws sws_converter;
};

}  // namespace p10::media