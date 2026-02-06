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
        return Ok(swr_);
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
    return Ok(swr_);
}

P10Error FfmpegSwr::transform(const AVFrame* source_frame, AVFrame** output_frame) {
    *output_frame = nullptr;

    auto target_frame = UniqueAVFrame(av_frame_alloc());
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
        return FfmpegWrapError(ret, "Could not fill target frame arrays");
    }

    SwrContext* swrConvContext = nullptr;
    try {
        swrConvContext = get_swr_context(
            source_frame->ch_layout,
            AVSampleFormat(source_frame->format),
            source_frame->sample_rate
        );
    } catch (const AvException&) {
        av_frame_free(&target_frame);
        throw;
    }
    const uint8_t** inData = (const uint8_t**)source_frame->data;
    P10_RETURN_IF_ERROR(wrap_ffmpeg_error(swr_convert(
        swrConvContext,
        target_frame->data,
        target_frame->nb_samples,
        inData,
        source_frame->nb_samples
    )));

    *output_frame = target_frame.release();
    return Ok(target_frame.release());
}

AudioFrame FfmpegSwr::resample(const AudioFrame& sourceFrame) {
    AudioFrame targetFrame(
        target_channel_layout_.nb_channels,
        int((sourceFrame.sampleSize() * size_t(target_sample_rate_))
            / size_t(sourceFrame.sampleRate())),
        toDtype(target_sample_format_),
        target_sample_rate_
    );
    AVChannelLayout source_channel_layout;
    av_channel_layout_default(&source_channel_layout, int(sourceFrame.channels()));
    SwrContext* swrConvContext = get_swr_context(
        source_channel_layout,
        toAVSampleFormat(sourceFrame.type()),
        sourceFrame.sampleRate()
    );

    std::array<const uint8_t*, 8> inPlanes;
    std::array<uint8_t*, 8> outPlanes;
    for (size_t i = 0; i < sourceFrame.channels(); ++i) {
        inPlanes[i] = sourceFrame.data<uint8_t>(i);
        outPlanes[i] = targetFrame.data<uint8_t>(i);
    }

    FfmpegExpect(
        swr_convert(
            swrConvContext,
            outPlanes.data(),
            int(targetFrame.sampleSize()),
            inPlanes.data(),
            int(sourceFrame.sampleSize())
        ),
        "Could not resample audio frame: "
    );

    return targetFrame;
}

AVFrame* FfmpegSwr::operator()(const AudioFrame& inFrame) {
    UniqueAVFrame avInFrame(asAVFrame(inFrame));
    return operator()(avInFrame.get());
}

}  // namespace p10::media