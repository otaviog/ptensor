#include "ffmpeg_video_decoder.hpp"

#include <libavcodec/codec_id.h>

#include "../video_queue.hpp"
#include "ffmpeg_memory.hpp"
#include "ffmpeg_wrap_error.hpp"

namespace p10::media {

namespace {
    VideoCodec video_codec_from_avcodec_id(AVCodecID codec_id) {
        switch (codec_id) {
            case AV_CODEC_ID_H264:
                return VideoCodec::H264;
            default:
                return VideoCodec(avcodec_get_name(codec_id));
        }
    }
}  // namespace

P10Result<FfmpegVideoDecoder::DecodeStatus>
FfmpegVideoDecoder::decode_packet(const AVPacket* pkt, VideoQueue& queue) {
    int ret = avcodec_send_packet(codec_ctx_, pkt);
    if (ret < 0 && ret != AVERROR_EOF) {
        return Err(wrap_ffmpeg_error(ret));
    }

    while (true) {
        UniqueAvFrame frame(av_frame_alloc());
        ret = avcodec_receive_frame(codec_ctx_, frame.get());
        if (ret == AVERROR(EAGAIN)) {
            return Ok(DecodeStatus::Again);
        } else if (ret == AVERROR_EOF) {
            return Ok(DecodeStatus::Eof);
        } else if (ret < 0) {
            return Err(wrap_ffmpeg_error(ret));
        }

        VideoFrame new_frame;
        sws_converter.transform(frame.get(), new_frame);
        new_frame.update_time(
            Time {Rational {stream_->time_base.num, stream_->time_base.den}, frame.get()->pts}
        );
        if (queue.emplace(std::move(new_frame)) == VideoQueue::Cancelled) {
            return Ok(DecodeStatus::Cancelled);
        }
    }
}

VideoParameters FfmpegVideoDecoder::get_video_parameters() const {
    VideoParameters params;
    auto codec = stream_->codecpar;
    params.width(codec->width)
        .height(codec->height)
        .codec(video_codec_from_avcodec_id(codec->codec_id))
        .bit_rate(codec->bit_rate);
    if (stream_->r_frame_rate.den != 0) {
        params.frame_rate(Rational {stream_->r_frame_rate.num, stream_->r_frame_rate.den});
    }

    return params;
}
}  // namespace p10::media
