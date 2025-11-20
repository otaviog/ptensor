#pragma once

extern "C" {
#include <libavcodec/packet.h>
#include <libavutil/frame.h>
#include <libswscale/swscale.h>
}
#include <ptensor/p10_error.hpp>

#include "video_frame.hpp"

namespace p10::media {
class FfmpegSws {
  public:
    FfmpegSws() = default;
    ~FfmpegSws();

    P10Error transform(const AVFrame* frame, VideoFrame& output_frame);

  private:
    SwsContext* get_sws_context(const AVFrame* frame);

    SwsContext* m_swsConvContext = nullptr;
};
}  // namespace p10::media