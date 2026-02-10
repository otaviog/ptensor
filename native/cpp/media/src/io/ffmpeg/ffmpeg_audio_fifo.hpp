#pragma once

#include "ptensor/p10_error.hpp"
extern "C" {
#include <libavutil/channel_layout.h>
#include <libavutil/frame.h>
}

struct AVAudioFifo;

namespace p10::media {

class AudioFrame;

class FfmpegAudioFifo {
  public:
    FfmpegAudioFifo();

    ~FfmpegAudioFifo();

    // Non-copyable
    FfmpegAudioFifo(const FfmpegAudioFifo&) = delete;
    FfmpegAudioFifo& operator=(const FfmpegAudioFifo&) = delete;

    // Movable
    FfmpegAudioFifo(FfmpegAudioFifo&& other) noexcept;
    FfmpegAudioFifo& operator=(FfmpegAudioFifo&& other) noexcept;

    void reset();

    void reset(AVChannelLayout channel_layout, AVSampleFormat sample_format, int sample_rate);

    P10Error add_samples(AVFrame* frame);

    P10Error add_samples(AudioFrame& frame);

    P10Error pop_samples(int frame_size, AVFrame** out_frame);

    bool empty() const;

    size_t num_samples() const;

    int sample_rate() const {
        return sample_rate_;
    }

    void clear();

  private:
    P10Error add_samples(void** data, int nb_samples, int sample_rate);
    AVAudioFifo* get_fifo();

    AVAudioFifo* audio_fifo_ = nullptr;
    AVChannelLayout channel_layout_;
    AVSampleFormat sample_format_ = AV_SAMPLE_FMT_NONE;
    int sample_rate_ = 0;
};
}  // namespace p10::media
