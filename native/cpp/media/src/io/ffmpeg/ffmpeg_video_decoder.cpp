#include "ffmpeg_video_decoder.hpp"

#include "ffmpeg_memory.hpp"
#include "ffmpeg_wrap_error.hpp"

namespace p10::media {

P10Error FfmpegVideoDecoder::decode_packet(const AVPacket* pkt, VideoFrame& out_frame) {
    while (true) {
        UniqueAvFrame frame(av_frame_alloc());
        int ret = avcodec_send_packet(codec_ctx_, pkt);
        if (ret < 0) {
            return wrap_ffmpeg_error(ret);
        }

        ret = avcodec_receive_frame(codec_ctx_, frame.get());
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            return wrap_ffmpeg_error(ret);
        }

        return sws_converter.transform(frame.get(), out_frame);
    }

    return P10Error::Ok;
}

VideoParameters FfmpegVideoDecoder::get_video_parameters() const {
    VideoParameters params;
    params.width(stream_->codecpar->width).height(stream_->codecpar->height);
    if (stream_->r_frame_rate.den != 0) {
        params.frame_rate(Rational {stream_->r_frame_rate.num, stream_->r_frame_rate.den});
    }
    return params;
}

}  // namespace p10::media
