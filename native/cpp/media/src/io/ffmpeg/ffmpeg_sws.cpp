#include "ffmpeg_sws.hpp"
#include "ffmpeg_wrap_error.hpp"

namespace p10::media {

FfmpegSws::~FfmpegSws() {
    sws_freeContext(m_swsConvContext);
    m_swsConvContext = nullptr;
}

P10Error FfmpegSws::transform(const AVFrame* frame, VideoFrame& output_frame) {
    SwsContext* sws_ctx = get_sws_context(frame);
    if (!sws_ctx) {
        return P10Error(P10Error::UnknownError, "Failed to get SwsContext");
    }

    uint8_t* dst_data[1] = {output_frame.as_bytes().data()};
    int dst_linesize[1] = {int(output_frame.width()) * 3};

    return wrap_error(
        sws_scale(
            sws_ctx,
            frame->data,
            frame->linesize,
            0,
            frame->height,
            dst_data,
            dst_linesize
        ),
        "sws_scale failed"
    );
}

SwsContext* FfmpegSws::get_sws_context(const AVFrame* frame) {
    m_swsConvContext = sws_getCachedContext(
        m_swsConvContext,
        frame->width,
        frame->height,
        static_cast<AVPixelFormat>(frame->format),
        frame->width,
        frame->height,
        AV_PIX_FMT_RGB24,
        SWS_POINT,
        nullptr,
        nullptr,
        nullptr
    );
    return m_swsConvContext;
}

}  // namespace p10::media
