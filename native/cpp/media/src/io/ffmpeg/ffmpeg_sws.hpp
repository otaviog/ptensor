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

    P10Error transform(const VideoFrame& frame, AVFrame** output_frame);

    std::optional<int> target_width() const {
        return target_width_;
    }

    std::optional<int> target_height() const {
        return target_height_;
    }

    void set_target_size(int width, int height) {
        target_width_ = width;
        target_height_ = height;
    }

    void reset_target_size() {
        target_width_ = std::nullopt;
        target_height_ = std::nullopt;
    }

    void set_target_pixel_format(AVPixelFormat format) {
        target_pixel_format_ = format;
    }

    AVPixelFormat target_pixel_format() const {
        return target_pixel_format_;
    }

  private:
    struct TargetSwsContextKey {
        int source_width;
        int source_height;
        AVPixelFormat source_format;
        int target_width;
        int target_height;
    };

    TargetSwsContextKey get_target_sws_context_key(const VideoFrame& src) const;
    TargetSwsContextKey get_target_sws_context_key(const AVFrame* src) const;

    SwsContext* get_sws_context(TargetSwsContextKey key);
    SwsContext* sws_context_ = nullptr;

    AVPixelFormat target_pixel_format_ = AV_PIX_FMT_RGB24;
    std::optional<int> target_width_;
    std::optional<int> target_height_;
};
}  // namespace p10::media