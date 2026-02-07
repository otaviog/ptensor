#pragma once

extern "C" {
#include <libavutil/channel_layout.h>
#include <libavutil/frame.h>
}

struct AVAudioFifo;

namespace p10::media {

class FfmpegAudioFifo {
  public:
    FfmpegAudioFifo();

    FfmpegAudioFifo(AVChannelLayout channel_layout, AVSampleFormat sample_format, int sample_rate);

    ~FfmpegAudioFifo();

    // Non-copyable
    FfmpegAudioFifo(const FfmpegAudioFifo&) = delete;
    FfmpegAudioFifo& operator=(const FfmpegAudioFifo&) = delete;

    // Movable
    FfmpegAudioFifo(FfmpegAudioFifo&& other) noexcept;
    FfmpegAudioFifo& operator=(FfmpegAudioFifo&& other) noexcept;

    void add_samples(AVFrame* frame);
    void add_zeros(int pad_size);
    AVFrame* pop_samples(int frame_size);
    bool empty() const;
    int num_samples() const;

    int sample_rate() const {
        return sample_rate_;
    }

    void clear();

  private:
    AVAudioFifo* get_fifo();

    AVAudioFifo* audio_fifo_ = nullptr;
    AVChannelLayout channel_layout_;
    AVSampleFormat sample_format_ = AV_SAMPLE_FMT_NONE;
    int sample_rate_ = 0;
};
}  // namespace p10::media
