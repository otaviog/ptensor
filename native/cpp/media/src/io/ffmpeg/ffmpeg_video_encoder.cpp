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

P10Error
FfmpegVideoEncoder::create(const VideoParameters& video_params, AVFormatContext* output_format) {
    const auto* codec = avcodec_find_encoder(codec_id_from_video_parameters(video_params));
    video_stream_ = avformat_new_stream(output_format, codec);
    if (video_stream_ == nullptr) {
        return P10Error::InvalidOperation << "Could not add video stream";
    }

    video_stream_->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
    video_encoder_context_ = avcodec_alloc_context3(codec);

    // Quality
    video_encoder_context_->bit_rate = video_params.bit_rate();
    video_encoder_context_->rc_buffer_size = 4 * 1000 * 1000;
    video_encoder_context_->rc_max_rate = 2 * 1000 * 1000;
    video_encoder_context_->rc_min_rate = int64_t(2.5 * 1000.0 * 1000.0);
    video_encoder_context_->gop_size = 12;
    av_opt_set(video_encoder_context_->priv_data, "preset", "slower", 0);

    // Imagining
    video_encoder_context_->pix_fmt = AV_PIX_FMT_YUV420P;
    video_encoder_context_->width = video_params.width();
    video_encoder_context_->height = video_params.height();

    P10_RETURN_IF_ERROR(wrap_ffmpeg_error(avcodec_open2(video_encoder_context_, codec, nullptr)));
    P10_RETURN_IF_ERROR(wrap_ffmpeg_error(
        avcodec_parameters_from_context(video_stream_->codecpar, video_encoder_context_)
    ));

    video_rescaler_.set_target_size(video_params.width(), video_params.height());
}
}  // namespace p10::media