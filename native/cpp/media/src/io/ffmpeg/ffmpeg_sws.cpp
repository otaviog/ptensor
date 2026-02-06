#include "ffmpeg_sws.hpp"

#include <libavutil/frame.h>

#include "ffmpeg_wrap_error.hpp"
#include "video_frame.hpp"

namespace p10::media {

FfmpegSws::~FfmpegSws() {
    sws_freeContext(sws_context_);
    sws_context_ = nullptr;
}

P10Error FfmpegSws::transform(const AVFrame* src, VideoFrame& dst) {
    auto sws_key = get_target_sws_context_key(src);
    SwsContext* sws_ctx = get_sws_context(sws_key);
    if (!sws_ctx) {
        return P10Error::InvalidOperation << "Failed to get SwsContext";
    }

    dst.create(sws_key.target_width, sws_key.target_height, PixelFormat::RGB24);

    uint8_t* dst_data[1] = {dst.as_bytes().data()};
    int dst_linesize[1] = {int(dst.stride().byte_stride(dst.dtype(), 0).unwrap())};

    int result =
        sws_scale(sws_ctx, src->data, src->linesize, 0, src->height, dst_data, dst_linesize);

    if (result <= 0) {
        return wrap_ffmpeg_error(result, "sws_scale failed");
    }
    return P10Error::Ok;
}

P10Error FfmpegSws::transform(const VideoFrame& src, AVFrame** dst) {
    auto sws_key = get_target_sws_context_key(src);

    SwsContext* sws_ctx = get_sws_context(sws_key);
    if (!sws_ctx) {
        return P10Error::InvalidOperation << "Failed to get SwsContext";
    }

    AVFrame* dst_frame = av_frame_alloc();
    dst_frame->width = sws_key.target_width;
    dst_frame->height = sws_key.target_height;
    dst_frame->format = target_pixel_format_;

    const auto targetBufferSize = av_image_get_buffer_size(
        target_pixel_format_,
        sws_key.target_width,
        sws_key.target_height,
        1
    );
    auto* targetBuffer = (uint8_t*)av_malloc(targetBufferSize);
    av_image_fill_arrays(
        dst_frame->data,
        dst_frame->linesize,
        targetBuffer,
        target_pixel_format_,
        sws_key.target_width,
        sws_key.target_height,
        1
    );

    const auto& src_bytes = src.as_bytes();
    uint8_t* src_data[1] = {const_cast<uint8_t*>(src_bytes.data())};
    int src_linesize[1] = {int(src.stride().byte_stride(src.dtype(), 0).unwrap())};

    int result = sws_scale(
        sws_ctx,
        src_data,
        src_linesize,
        0,
        dst_frame->height,
        dst_frame->data,
        dst_frame->linesize
    );

    if (result <= 0) {
        return wrap_ffmpeg_error(result, "sws_scale failed");
    }
    return P10Error::Ok;
}

FfmpegSws::TargetSwsContextKey FfmpegSws::get_target_sws_context_key(const VideoFrame& src) const {
    TargetSwsContextKey key;
    key.source_width = static_cast<int>(src.width());
    key.source_height = static_cast<int>(src.height());
    key.source_format = AV_PIX_FMT_RGB24;
    key.target_width = target_width_.has_value() ? *target_width_ : key.source_width;
    key.target_height = target_height_.has_value() ? *target_height_ : key.source_height;
    return key;
}

FfmpegSws::TargetSwsContextKey FfmpegSws::get_target_sws_context_key(const AVFrame* src) const {
    TargetSwsContextKey key;
    key.source_width = src->width;
    key.source_height = src->height;
    key.source_format = static_cast<AVPixelFormat>(src->format);
    key.target_width = target_width_.has_value() ? *target_width_ : key.source_width;
    key.target_height = target_height_.has_value() ? *target_height_ : key.source_height;
    return key;
}

SwsContext* FfmpegSws::get_sws_context(TargetSwsContextKey key) {
    sws_context_ = sws_getCachedContext(
        sws_context_,
        key.source_width,
        key.source_height,
        key.source_format,
        key.target_width,
        key.target_height,
        target_pixel_format_,
        SWS_POINT,
        nullptr,
        nullptr,
        nullptr
    );
    return sws_context_;
}

}  // namespace p10::media
