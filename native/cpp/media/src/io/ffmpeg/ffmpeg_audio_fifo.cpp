#include "ffmpeg_audio_fifo.hpp"

#include "ffmpeg_wrap_error.hpp"
#include "ptensor/p10_error.hpp"

extern "C" {
#include <libavutil/audio_fifo.h>
#include <libavutil/mem.h>
}

#include "audio_frame.hpp"

namespace p10::media {

namespace {
    Dtype av_sample_fmt_to_dtype(AVSampleFormat sample_fmt) {
        switch (sample_fmt) {
            case AV_SAMPLE_FMT_U8:
                return Dtype::Uint8;
            case AV_SAMPLE_FMT_S16:
                return Dtype::Int16;
            case AV_SAMPLE_FMT_S32:
                return Dtype::Int32;
            case AV_SAMPLE_FMT_FLT:
                return Dtype::Float32;
            case AV_SAMPLE_FMT_DBL:
                return Dtype::Float64;
            case AV_SAMPLE_FMT_U8P:
                return Dtype::Uint8;
            case AV_SAMPLE_FMT_S16P:
                return Dtype::Int16;
            case AV_SAMPLE_FMT_S32P:
                return Dtype::Int32;
            case AV_SAMPLE_FMT_FLTP:
                return Dtype::Float32;
            case AV_SAMPLE_FMT_DBLP:
                return Dtype::Float64;
            default:
                throw std::runtime_error(
                    "Unsupported sample format: " + std::to_string(sample_fmt)
                );
        }
    }
}  // namespace

FfmpegAudioFifo::FfmpegAudioFifo() {
    av_channel_layout_default(&channel_layout_, 2);
}

FfmpegAudioFifo::~FfmpegAudioFifo() {
    if (audio_fifo_ != nullptr) {
        av_audio_fifo_free(audio_fifo_);
        audio_fifo_ = nullptr;
    }
}

FfmpegAudioFifo::FfmpegAudioFifo(FfmpegAudioFifo&& other) noexcept :
    audio_fifo_(other.audio_fifo_),
    channel_layout_(other.channel_layout_),
    sample_format_(other.sample_format_),
    sample_rate_(other.sample_rate_) {
    other.audio_fifo_ = nullptr;
}

FfmpegAudioFifo& FfmpegAudioFifo::operator=(FfmpegAudioFifo&& other) noexcept {
    if (this != &other) {
        if (audio_fifo_ != nullptr) {
            av_audio_fifo_free(audio_fifo_);
        }
        audio_fifo_ = other.audio_fifo_;
        channel_layout_ = other.channel_layout_;
        sample_format_ = other.sample_format_;
        sample_rate_ = other.sample_rate_;
        other.audio_fifo_ = nullptr;
    }
    return *this;
}

void FfmpegAudioFifo::reset() {
    if (audio_fifo_ != nullptr) {
        av_audio_fifo_free(audio_fifo_);
        audio_fifo_ = nullptr;
    }
    av_channel_layout_default(&channel_layout_, 2);
    sample_format_ = AV_SAMPLE_FMT_NONE;
    sample_rate_ = 0;
}

void FfmpegAudioFifo::reset(
    AVChannelLayout channel_layout,
    AVSampleFormat sample_format,
    int sample_rate
) {
    channel_layout_ = channel_layout;
    sample_format_ = sample_format;
    sample_rate_ = sample_rate;
    if (audio_fifo_ != nullptr) {
        av_audio_fifo_free(audio_fifo_);
        audio_fifo_ = nullptr;
    }
}

P10Error FfmpegAudioFifo::add_samples(AVFrame* frame) {
    return add_samples(
        reinterpret_cast<void**>(frame->data),
        frame->nb_samples,
        frame->sample_rate
    );
}

P10Error FfmpegAudioFifo::add_samples(AudioFrame& frame) {
    const Dtype dtype = frame.samples().dtype();
    if (av_sample_fmt_to_dtype(sample_format_) != dtype) {
        return P10Error::InvalidArgument << "Sample format mismatch: expected "
            + std::to_string(sample_format_) + ", got " + std::to_string(frame.samples().dtype());
    }

    std::array<void*, AV_NUM_DATA_POINTERS> data = {nullptr};

    auto samples_start = frame.samples().as_bytes().data();
    assert(samples_start != nullptr);

    const size_t elem_size = dtype.size_bytes();
    for (size_t i = 0; i < std::min(data.size(), static_cast<size_t>(frame.channels_count()));
         ++i) {
        data[i] = reinterpret_cast<void*>(samples_start + i * frame.samples_count() * elem_size);
    }

    return add_samples(data.data(), frame.samples_count(), frame.sample_rate());
}

P10Error FfmpegAudioFifo::add_samples(void** data, int nb_samples, int sample_rate) {
    if (sample_rate != sample_rate_) {
        return P10Error::InvalidArgument << "Sample rate mismatch: expected "
            + std::to_string(sample_rate_) + ", got " + std::to_string(sample_rate);
    }
    AVAudioFifo* fifo = get_fifo();

    assert(fifo != nullptr);
    P10_RETURN_IF_ERROR(wrap_ffmpeg_error(av_audio_fifo_write(fifo, data, nb_samples)));
    return P10Error::Ok;
}

P10Error FfmpegAudioFifo::pop_samples(int frame_size, AVFrame** out_frame) {
    AVAudioFifo* fifo = get_fifo();
    if (fifo == nullptr || av_audio_fifo_size(fifo) < frame_size) {
        return P10Error::InvalidArgument << "Not enough samples in FIFO";
    }

    AVFrame* frame = av_frame_alloc();
    frame->nb_samples = frame_size;
    frame->format = sample_format_;
    frame->ch_layout = channel_layout_;
    frame->sample_rate = sample_rate_;

    av_frame_get_buffer(frame, 0);
    av_audio_fifo_read(fifo, reinterpret_cast<void**>(frame->data), frame_size);

    *out_frame = frame;
    return P10Error::Ok;
}

bool FfmpegAudioFifo::empty() const {
    if (audio_fifo_ == nullptr) {
        return true;
    }
    return av_audio_fifo_size(audio_fifo_) == 0;
}

size_t FfmpegAudioFifo::num_samples() const {
    if (audio_fifo_ == nullptr) {
        return 0;
    }
    return av_audio_fifo_size(audio_fifo_);
}

void FfmpegAudioFifo::clear() {
    if (audio_fifo_ != nullptr) {
        av_audio_fifo_reset(audio_fifo_);
    }
}

AVAudioFifo* FfmpegAudioFifo::get_fifo() {
    if (audio_fifo_ == nullptr) {
        audio_fifo_ = av_audio_fifo_alloc(sample_format_, channel_layout_.nb_channels, 1);
    }
    return audio_fifo_;
}

}  // namespace p10::media