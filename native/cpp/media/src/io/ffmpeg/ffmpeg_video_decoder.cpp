#include "ffmpeg_video_decoder.hpp"

#include "../video_queue.hpp"
#include "ffmpeg_memory.hpp"
#include "ffmpeg_wrap_error.hpp"

namespace p10::media {

P10Result<FfmpegVideoDecoder::DecodeStatus>
FfmpegVideoDecoder::decode_packet(const AVPacket* pkt, VideoQueue& queue) {
    VideoFrame new_frame;
    while (true) {
        UniqueAvFrame frame(av_frame_alloc());
        int ret = avcodec_send_packet(codec_ctx_, pkt);
        if (ret < 0) {
            return Err(wrap_ffmpeg_error(ret));
        }

        ret = avcodec_receive_frame(codec_ctx_, frame.get());
        if (ret == AVERROR(EAGAIN)) {
            return Ok(DecodeStatus::Again);
        } else if (ret == AVERROR_EOF) {
            return Ok(DecodeStatus::Eof);
        } else if (ret < 0) {
            return Err(wrap_ffmpeg_error(ret));
        }

        sws_converter.transform(frame.get(), new_frame);
        if (queue.emplace(std::move(new_frame)) == VideoQueue::Cancelled) {
            return Ok(DecodeStatus::Cancelled);
        }
    }

    return Ok(DecodeStatus::FrameDecoded);
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
