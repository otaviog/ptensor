#include "ffmpeg_audio_fifo.hpp"

extern "C" {
#include <libavutil/audio_fifo.h>
#include <libavutil/mem.h>
}

namespace p10::media {

FfmpegAudioFifo::FfmpegAudioFifo() {
    av_channel_layout_default(&channel_layout_, 2);
}

FfmpegAudioFifo::FfmpegAudioFifo(
    AVChannelLayout channel_layout,
    AVSampleFormat sample_format,
    int sample_rate
) :
    channel_layout_(channel_layout),
    sample_format_(sample_format),
    sample_rate_(sample_rate) {}

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

void FfmpegAudioFifo::add_samples(AVFrame* frame) {
    AVAudioFifo* fifo = get_fifo();
    if (fifo != nullptr) {
        av_audio_fifo_write(fifo, reinterpret_cast<void**>(frame->data), frame->nb_samples);
    }
}

void FfmpegAudioFifo::add_zeros(int pad_size) {
    AVAudioFifo* fifo = get_fifo();
    if (fifo != nullptr) {
        // Allocate zero-filled buffer
        int channels = channel_layout_.nb_channels;
        int bytes_per_sample = av_get_bytes_per_sample(sample_format_);
        size_t buffer_size = static_cast<size_t>(pad_size * channels * bytes_per_sample);
        auto* zeros = static_cast<uint8_t*>(av_mallocz(buffer_size));

        uint8_t* data[8] = {zeros, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        av_audio_fifo_write(fifo, reinterpret_cast<void**>(data), pad_size);
        av_free(zeros);
    }
}

AVFrame* FfmpegAudioFifo::pop_samples(int frame_size) {
    AVAudioFifo* fifo = get_fifo();
    if (fifo == nullptr || av_audio_fifo_size(fifo) < frame_size) {
        return nullptr;
    }

    AVFrame* frame = av_frame_alloc();
    frame->nb_samples = frame_size;
    frame->format = sample_format_;
    frame->ch_layout = channel_layout_;
    frame->sample_rate = sample_rate_;

    av_frame_get_buffer(frame, 0);
    av_audio_fifo_read(fifo, reinterpret_cast<void**>(frame->data), frame_size);

    return frame;
}

bool FfmpegAudioFifo::empty() const {
    if (audio_fifo_ == nullptr) {
        return true;
    }
    return av_audio_fifo_size(audio_fifo_) == 0;
}

int FfmpegAudioFifo::num_samples() const {
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