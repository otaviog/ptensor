#include "ffmpeg_swr.hpp"

#include <array>

#include "audio_frame.hpp"
#include "ptensor/p10_error.hpp"

extern "C" {
#include <libavutil/channel_layout.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
}

#include "ffmpeg_memory.hpp"
#include "ffmpeg_wrap_error.hpp"

namespace p10::media {

void FfmpegSwr::reset(
    AVChannelLayout target_channel_layout,
    AVSampleFormat target_sample_format,
    int target_sample_rate
) {
    release();
    target_channel_layout_ = target_channel_layout;
    target_sample_format_ = target_sample_format;
    target_sample_rate_ = target_sample_rate;
}

void FfmpegSwr::release() {
    swr_free(&swr_);
    swr_ = nullptr;
}

P10Result<SwrContext*> FfmpegSwr::get_swr_context(
    AVChannelLayout source_channel_layout,
    AVSampleFormat source_sample_format,
    int source_sample_rate
) {
    if (swr_ != nullptr) {
        SwrContext* result = swr_;
        return Ok(std::move(result));
    }

    P10_RETURN_ERR_IF_ERROR(wrap_ffmpeg_error(swr_alloc_set_opts2(
        &swr_,
        &target_channel_layout_,
        target_sample_format_,
        target_sample_rate_,
        &source_channel_layout,
        source_sample_format,
        source_sample_rate,
        0,
        nullptr
    )));

    P10_RETURN_ERR_IF_ERROR(wrap_ffmpeg_error(swr_init(swr_)));
    SwrContext* result = swr_;
    return Ok(std::move(result));
}

P10Error FfmpegSwr::transform(const AVFrame* source_frame, AVFrame** output_frame) {
    *output_frame = nullptr;

    AVFrame* target_frame = av_frame_alloc();
    target_frame->sample_rate = target_sample_rate_;
    target_frame->format = target_sample_format_;
    target_frame->nb_samples =
        int((source_frame->nb_samples * size_t(target_sample_rate_))
            / size_t(source_frame->sample_rate));
    target_frame->ch_layout = target_channel_layout_;

    const auto targetBufferSize = av_samples_get_buffer_size(
        nullptr,
        target_frame->ch_layout.nb_channels,
        target_frame->nb_samples,
        target_sample_format_,
        1
    );
    if (targetBufferSize < 0) {
        av_frame_free(&target_frame);
        return P10Error::InvalidOperation
            << "Could not get target buffer size: " + std::to_string(targetBufferSize);
    }

    auto* targetBuffer = (uint8_t*)av_malloc(targetBufferSize);
    auto ret = av_samples_fill_arrays(
        target_frame->data,
        target_frame->linesize,
        targetBuffer,
        target_frame->ch_layout.nb_channels,
        target_frame->nb_samples,
        target_sample_format_,
        1
    );
    if (ret < 0) {
        av_frame_free(&target_frame);
        return wrap_ffmpeg_error(ret, "Could not fill target frame arrays");
    }

    auto swr_result = get_swr_context(
        source_frame->ch_layout,
        AVSampleFormat(source_frame->format),
        source_frame->sample_rate
    );
    if (swr_result.is_error()) {
        av_frame_free(&target_frame);
        return swr_result.error();
    }
    SwrContext* swrConvContext = swr_result.unwrap();

    const uint8_t** inData = (const uint8_t**)source_frame->data;
    int convert_result = swr_convert(
        swrConvContext,
        target_frame->data,
        target_frame->nb_samples,
        inData,
        source_frame->nb_samples
    );
    if (convert_result < 0) {
        av_frame_free(&target_frame);
        return wrap_ffmpeg_error(convert_result, "Could not convert audio samples");
    }

    *output_frame = target_frame;
    return P10Error::Ok;
}

}  // namespace p10::media