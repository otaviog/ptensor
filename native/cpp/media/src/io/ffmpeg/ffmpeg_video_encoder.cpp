#include "ffmpeg_video_encoder.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/codec_id.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/opt.h>
#include <libavutil/time.h>
}

#include "ffmpeg_sws.hpp"
#include "ffmpeg_wrap_error.hpp"
#include "ptensor/p10_error.hpp"
#include "video_frame.hpp"
#include "video_parameters.hpp"

namespace p10::media {
namespace {
AVCodecID codec_id_from_video_parameters(const VideoParameters& video_params) {
    switch (video_params.codec().type()) {
        case VideoCodec::CodecType::H264:
            return AV_CODEC_ID_H264;
        default:
            return AV_CODEC_ID_NONE;
    }
}

}  // namespace

FfmpegVideoEncoder::~FfmpegVideoEncoder() {
    while (!packet_queue_.empty()) {
        AVPacket* pkt = packet_queue_.front();
        packet_queue_.pop();
        av_packet_free(&pkt);
    }
    if (video_encoder_context_ != nullptr) {
        avcodec_free_context(&video_encoder_context_);
    }
}

P10Error FfmpegVideoEncoder::create(
    const VideoParameters& video_params,
    AVFormatContext* output_format
) {
    AVCodecID codec_id = codec_id_from_video_parameters(video_params);
    if (codec_id == AV_CODEC_ID_NONE) {
        // Default to H264 if no codec specified
        codec_id = AV_CODEC_ID_H264;
    }

    const AVCodec* codec = avcodec_find_encoder(codec_id);
    if (codec == nullptr) {
        return P10Error::IoError << "Could not find video encoder for codec";
    }

    video_stream_ = avformat_new_stream(output_format, codec);
    if (video_stream_ == nullptr) {
        return P10Error::InvalidOperation << "Could not add video stream";
    }

    video_stream_->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
    video_encoder_context_ = avcodec_alloc_context3(codec);
    if (video_encoder_context_ == nullptr) {
        return P10Error::OutOfMemory << "Could not allocate video codec context";
    }

    // Quality settings
    video_encoder_context_->bit_rate = video_params.bit_rate();
    video_encoder_context_->rc_buffer_size = 4 * 1000 * 1000;
    video_encoder_context_->rc_max_rate = 2 * 1000 * 1000;
    video_encoder_context_->rc_min_rate = int64_t(2.5 * 1000.0 * 1000.0);
    video_encoder_context_->gop_size = 12;
    av_opt_set(video_encoder_context_->priv_data, "preset", "medium", 0);

    // Image format
    video_encoder_context_->pix_fmt = AV_PIX_FMT_YUV420P;
    video_encoder_context_->width = video_params.width();
    video_encoder_context_->height = video_params.height();

    // Time base - use frame rate
    Rational frame_rate = video_params.frame_rate();
    video_encoder_context_->time_base = {
        static_cast<int>(frame_rate.num()),
        static_cast<int>(frame_rate.den())
    };
    video_stream_->time_base = video_encoder_context_->time_base;

    // Some formats require global headers
    if (output_format->oformat->flags & AVFMT_GLOBALHEADER) {
        video_encoder_context_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    P10_RETURN_IF_ERROR(wrap_ffmpeg_error(avcodec_open2(video_encoder_context_, codec, nullptr)));
    P10_RETURN_IF_ERROR(wrap_ffmpeg_error(
        avcodec_parameters_from_context(video_stream_->codecpar, video_encoder_context_)
    ));

    video_rescaler_.set_target_size(video_params.width(), video_params.height());
    video_rescaler_.set_target_pixel_format(AV_PIX_FMT_YUV420P);

    return P10Error::Ok;
}

P10Error FfmpegVideoEncoder::encode_frame(const VideoFrame& frame, int64_t pts) {
    AVFrame* av_frame = nullptr;
    P10_RETURN_IF_ERROR(video_rescaler_.transform(frame, &av_frame));

    av_frame->pts = pts;
    av_frame->pkt_dts = pts;

    P10Error err = send_frame(av_frame);
    av_frame_free(&av_frame);

    if (err.is_error()) {
        return err;
    }

    return receive_packets();
}

P10Error FfmpegVideoEncoder::flush() {
    // Send NULL frame to signal end of stream
    int ret = avcodec_send_frame(video_encoder_context_, nullptr);
    if (ret < 0 && ret != AVERROR_EOF) {
        return wrap_ffmpeg_error(ret, "Failed to flush video encoder");
    }
    return receive_packets();
}

P10Error FfmpegVideoEncoder::send_frame(AVFrame* frame) {
    int ret = avcodec_send_frame(video_encoder_context_, frame);
    if (ret < 0) {
        return wrap_ffmpeg_error(ret, "Failed to send frame to video encoder");
    }
    return P10Error::Ok;
}

P10Error FfmpegVideoEncoder::receive_packets() {
    while (true) {
        AVPacket* pkt = av_packet_alloc();
        int ret = avcodec_receive_packet(video_encoder_context_, pkt);

        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            av_packet_free(&pkt);
            break;
        } else if (ret < 0) {
            av_packet_free(&pkt);
            return wrap_ffmpeg_error(ret, "Failed to receive packet from video encoder");
        }

        pkt->stream_index = video_stream_->index;
        packet_queue_.push(pkt);
    }
    return P10Error::Ok;
}

AVPacket* FfmpegVideoEncoder::pop_packet() {
    if (packet_queue_.empty()) {
        return nullptr;
    }
    AVPacket* pkt = packet_queue_.front();
    packet_queue_.pop();
    return pkt;
}

}  // namespace p10::media
