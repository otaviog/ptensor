#include "ffmpeg_audio_encoder.hpp"

#include "audio_frame.hpp"
#include "ffmpeg_memory.hpp"
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

namespace {
    AVCodecID codec_id_from_audio_parameters(AudioCodec codec) {
        switch (codec.type()) {
            case AudioCodec::CodecType::AAC:
                return AV_CODEC_ID_AAC;
            default:
                return AV_CODEC_ID_NONE;
        }
    }

    AVSampleFormat dtype_to_planar_av_sample_format(Dtype dtype) {
        switch (dtype.value) {
            case Dtype::Uint8:
                return AV_SAMPLE_FMT_U8P;
            case Dtype::Int16:
                return AV_SAMPLE_FMT_S16P;
            case Dtype::Int32:
                return AV_SAMPLE_FMT_S32P;
            case Dtype::Float32:
                return AV_SAMPLE_FMT_FLTP;
            case Dtype::Float64:
                return AV_SAMPLE_FMT_DBLP;
            default:
                return AV_SAMPLE_FMT_NONE;
        }
    }
}  // namespace

FfmpegAudioEncoder::~FfmpegAudioEncoder() {
    while (!encoded_packets_.empty()) {
        AVPacket* pkt = encoded_packets_.front();
        encoded_packets_.pop();
        av_packet_free(&pkt);
    }
    reset();
}

P10Error
FfmpegAudioEncoder::create(const AudioParameters& audio_params, AVFormatContext* format_ctx) {
    reset();

    AVCodecID codec_id = codec_id_from_audio_parameters(audio_params.codec());
    const auto* codec = avcodec_find_encoder(codec_id);
    if (codec == nullptr) {
        return P10Error::IoError << std::string("Could not find audio codec for codec ")
            + audio_params.codec().to_string();
    }

    stream_ = avformat_new_stream(format_ctx, codec);
    if (stream_ == nullptr) {
        return P10Error::InvalidOperation << "Could not add audio stream";
    }

    codec_context_ = avcodec_alloc_context3(codec);
    if (codec_context_ == nullptr) {
        return P10Error::OutOfMemory << "Could not allocate audio codec context";
    }

    codec_context_->bit_rate = static_cast<int64_t>(audio_params.bit_rate());
    codec_context_->sample_rate = static_cast<int>(audio_params.audio_sample_rate_hz());

    // TODO: Use any format instead of FLTP as default sample format
    codec_context_->sample_fmt = AV_SAMPLE_FMT_FLTP;
    av_channel_layout_default(
        &codec_context_->ch_layout,
        static_cast<int>(audio_params.audio_channels())
    );

    // Time base
    const AVRational time_base {1, static_cast<int>(audio_params.audio_sample_rate_hz())};
    stream_->time_base = time_base;
    codec_context_->time_base = time_base;
    codec_context_->strict_std_compliance = FF_COMPLIANCE_EXPERIMENTAL;
    if (format_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        codec_context_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
    P10_RETURN_IF_ERROR(wrap_ffmpeg_error(avcodec_open2(codec_context_, codec, nullptr)));

    P10_RETURN_IF_ERROR(
        wrap_ffmpeg_error(avcodec_parameters_from_context(stream_->codecpar, codec_context_))
    );

    encoding_fifo_
        .reset(codec_context_->ch_layout, codec_context_->sample_fmt, codec_context_->sample_rate);
    resampler_
        .reset(codec_context_->ch_layout, codec_context_->sample_fmt, codec_context_->sample_rate);

    return P10Error::Ok;
}

void FfmpegAudioEncoder::reset() {
    if (codec_context_ != nullptr) {
        avcodec_free_context(&codec_context_);
    }
    stream_ = nullptr;
}

P10Error FfmpegAudioEncoder::encode(const AudioFrame& frame) {
    AVSampleFormat input_fmt = dtype_to_planar_av_sample_format(frame.samples().dtype());
    if (input_fmt == AV_SAMPLE_FMT_NONE) {
        return P10Error::InvalidArgument << "Unsupported audio sample format for encoding";
    }

    // Build a temporary AVFrame referencing the AudioFrame's tensor data (zero-copy).
    // The const_cast is safe: the resampler's transform() takes const AVFrame* and
    // does not modify the source data.
    AVFrame* source_frame = av_frame_alloc();
    source_frame->nb_samples = static_cast<int>(frame.samples_count());
    source_frame->format = input_fmt;
    source_frame->sample_rate = static_cast<int>(frame.sample_rate());
    av_channel_layout_default(&source_frame->ch_layout, static_cast<int>(frame.channels_count()));

    const auto* samples_data = frame.samples().as_bytes().data();
    const size_t elem_size = frame.samples().dtype().size_bytes();
    for (int ch = 0; ch < source_frame->ch_layout.nb_channels; ++ch) {
        source_frame->data[ch] = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(
            samples_data
            + static_cast<ptrdiff_t>(ch) * static_cast<ptrdiff_t>(frame.samples_count())
                * static_cast<ptrdiff_t>(elem_size)
        ));
    }
    source_frame->linesize[0] =
        static_cast<int>(static_cast<size_t>(frame.samples_count()) * elem_size);

    // Resample to the codec's target format/rate/channels
    AVFrame* resampled = nullptr;
    P10Error resample_err = resampler_.transform(source_frame, &resampled);
    av_frame_free(&source_frame);
    P10_RETURN_IF_ERROR(resample_err);

    // Add resampled samples to the encoding FIFO
    P10Error fifo_err = encoding_fifo_.add_samples(resampled);
    av_frame_free(&resampled);
    P10_RETURN_IF_ERROR(fifo_err);

    return flush_encoding_fifo();
}

P10Error FfmpegAudioEncoder::flush() {
    // some codecs require have frame_size = 0, which means we can feed any number of samples. In that case, use a reasonable default frame size for flushing.
    const int codec_frame_size = codec_context_->frame_size > 0 ? codec_context_->frame_size : 1024;
    while (!encoding_fifo_.empty()) {
        int remaining = static_cast<int>(encoding_fifo_.num_samples());
        int frame_size = std::min(remaining, codec_frame_size);

        AVFrame* frame_to_encode = nullptr;
        P10_RETURN_IF_ERROR(encoding_fifo_.pop_samples(frame_size, &frame_to_encode));

        frame_to_encode->pts = audio_pts_;
        audio_pts_ += frame_to_encode->nb_samples;

        int ret = avcodec_send_frame(codec_context_, frame_to_encode);
        av_frame_free(&frame_to_encode);
        P10_RETURN_IF_ERROR(wrap_ffmpeg_error(ret));

        P10_RETURN_IF_ERROR(receive_packets());
    }

    // Send NULL frame to drain the encoder
    P10_RETURN_IF_ERROR(wrap_ffmpeg_error(avcodec_send_frame(codec_context_, nullptr)));
    return receive_packets();
}

P10Error FfmpegAudioEncoder::flush_encoding_fifo() {
    const int codec_frame_size = codec_context_->frame_size > 0 ? codec_context_->frame_size : 1024;
    while (static_cast<int>(encoding_fifo_.num_samples()) >= codec_frame_size) {
        AVFrame* frame_to_encode = nullptr;
        P10_RETURN_IF_ERROR(encoding_fifo_.pop_samples(codec_frame_size, &frame_to_encode)
        );

        frame_to_encode->pts = audio_pts_;
        audio_pts_ += frame_to_encode->nb_samples;

        int ret = avcodec_send_frame(codec_context_, frame_to_encode);
        av_frame_free(&frame_to_encode);
        P10_RETURN_IF_ERROR(wrap_ffmpeg_error(ret));

        P10_RETURN_IF_ERROR(receive_packets());
    }

    return P10Error::Ok;
}

P10Error FfmpegAudioEncoder::receive_packets() {
    while (true) {
        UniqueAvPacket pkt(av_packet_alloc());
        int ret = avcodec_receive_packet(codec_context_, pkt.get());

        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            return wrap_ffmpeg_error(ret, "Failed to receive packet from audio encoder");
        }

        pkt->stream_index = stream_->index;
        encoded_packets_.push(pkt.release());
    }
    return P10Error::Ok;
}

}  // namespace p10::media