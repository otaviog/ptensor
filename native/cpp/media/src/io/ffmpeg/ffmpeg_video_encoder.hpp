#pragma once

#include <queue>

#include "ffmpeg_sws.hpp"

struct AVFormatContext;
struct AVStream;
struct AVCodecContext;
struct AVPacket;

namespace p10::media {
class VideoParameters;
class VideoFrame;

class FfmpegVideoEncoder {
  public:
    FfmpegVideoEncoder() = default;
    ~FfmpegVideoEncoder();

    P10Error create(const VideoParameters& video_params, AVFormatContext* output_format);

    P10Error encode(const VideoFrame& frame, int64_t pts);

    P10Error flush();

    bool has_packets() const {
        return !packet_queue_.empty();
    }

    AVPacket* pop_encoded_packet();

    AVStream* stream() const {
        return stream_;
    }

    AVCodecContext* codec_context() const {
        return video_encoder_context_;
    }

  private:
    P10Error receive_packets();

    AVStream* stream_ = nullptr;
    AVCodecContext* video_encoder_context_ = nullptr;
    FfmpegSws video_rescaler_;

    std::queue<AVPacket*> packet_queue_;
};

}  // namespace p10::media
