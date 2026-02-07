#include "ffmpeg_audio_encoder.hpp"

#include "ffmpeg_wrap_error.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/codec_id.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/opt.h>
#include <libavutil/time.h>
}

#include "audio_parameters.hpp"
#include "ptensor/p10_error.hpp"

namespace p10::media {

FfmpegAudioEncoder::~FfmpegAudioEncoder() {
    if (codec_context_ != nullptr) {
        avcodec_free_context(&codec_context_);
    }
}

namespace {
    AVCodecID codec_id_from_audio_parameters(AudioCodec codec) {
        switch (codec.type()) {
            case AudioCodec::CodecType::AAC:
                return AV_CODEC_ID_AAC;
            default:
                return AV_CODEC_ID_NONE;
        }
    }
}

P10Error FfmpegAudioEncoder::create(const AudioParameters& audio_params, AVStream* audio_stream) {
    /******************************
     * Audio part
     ******************************/
    stream_ = audio_stream;
    const auto* audioCodec =
        avcodec_find_encoder(codec_id_from_audio_parameters(audio_params.codec()));
    if (audioCodec == nullptr) {
        return P10Error::IoError << std::string("Could not find audio codec for codec ")
            + audio_params.codec().to_string();
    }

    codec_context_ = avcodec_alloc_context3(audioCodec);
    codec_context_->bit_rate = static_cast<int64_t>(audio_params.bit_rate());
    codec_context_->sample_rate = static_cast<int>(audio_params.audio_sample_rate_hz());

    // Use FLTP as default sample format for AAC encoder
    codec_context_->sample_fmt = AV_SAMPLE_FMT_FLTP;
    av_channel_layout_default(
        &codec_context_->ch_layout,
        static_cast<int>(audio_params.audio_channels())
    );
    // Time
    const AVRational audioTimeBase {1, static_cast<int>(audio_params.audio_sample_rate_hz())};
    stream_->time_base = audioTimeBase;
    codec_context_->time_base = audioTimeBase;
    codec_context_->strict_std_compliance = FF_COMPLIANCE_EXPERIMENTAL;
    //if (m_outputFormatContext->oformat->flags & AVFMT_GLOBALHEADER)
    //    codec_context_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    P10_RETURN_IF_ERROR(wrap_ffmpeg_error(avcodec_open2(codec_context_, audioCodec, nullptr)));

    P10_RETURN_IF_ERROR(
        wrap_ffmpeg_error(avcodec_parameters_from_context(stream_->codecpar, codec_context_))
    );

    resampler_.reset(
        codec_context_->ch_layout,
        codec_context_->sample_fmt,
        codec_context_->sample_rate
    );

    return P10Error::Ok;
}

}  // namespace p10::media