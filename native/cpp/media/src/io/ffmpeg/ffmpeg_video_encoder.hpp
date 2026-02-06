#pragma once

#include "ffmpeg_sws.hpp"

struct AVFormatContext;
struct AVStream;
struct AVCodecContext;

namespace p10::media {
class VideoParameters;

struct FfmpegVideoEncoder {
P10Error create(const VideoParameters& video_params, AVFormatContext* output_format);

    AVStream* video_stream_ = nullptr;
    AVCodecContext* video_encoder_context_ = nullptr;
    FfmpegSws video_rescaler_;

    std::queue<AVFrame*> video_write_queue_;
};

}  // namespace p10::media