#pragma once

#include <ptensor/p10_result.hpp>

extern "C" {
#include <libavutil/channel_layout.h>
#include <libavutil/samplefmt.h>
}

struct SwrContext;
struct AVFrame;

namespace p10::media {
class AudioFrame;

class FfmpegSwr {
  public:
    FfmpegSwr() {
        av_channel_layout_default(&target_channel_layout_, 2);
    }

    FfmpegSwr(
        AVChannelLayout target_channel_layout,
        AVSampleFormat target_sample_format,
        int target_sample_rate
    ) :
        target_channel_layout_(target_channel_layout),
        target_sample_format_(target_sample_format),
        target_sample_rate_(target_sample_rate) {}

    ~FfmpegSwr() {
        release();
    }

    P10Error transform(const AVFrame* source_frame, AVFrame** output_frame);

    void reset(
        AVChannelLayout target_channel_layout,
        AVSampleFormat target_sample_format,
        int target_sample_rate
    );

  private:
    void release();
    P10Result<SwrContext*> get_swr_context(
        AVChannelLayout source_channel_layout,
        AVSampleFormat source_sample_format,
        int source_sample_rate
    );

    SwrContext* swr_ = nullptr;
    AVChannelLayout target_channel_layout_;
    AVSampleFormat target_sample_format_ = AV_SAMPLE_FMT_FLTP;
    int target_sample_rate_ = 48000;
};
}  // namespace p10::media