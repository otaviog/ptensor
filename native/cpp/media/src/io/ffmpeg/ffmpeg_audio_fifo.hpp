#pragma once

extern "C" {
#include <libavutil/audio_fifo.h>
#include <libavutil/channel_layout.h>
#include <libavutil/frame.h>
}
#include "Exception.hpp"
#include "Frame.hpp"

namespace p10::media {

class FfmpegAudioFifo {
  public:
    FfmpegAudioFifo() {
        av_channel_layout_default(&m_channelLayout, 2);
    }

    FfmpegAudioFifo(AVChannelLayout channelLayout, AVSampleFormat sampleFormat, int sampleRate) :
        m_channelLayout(channelLayout),
        m_sampleFormat(sampleFormat),
        m_sampleRate(sampleRate) {}

    ~FfmpegAudioFifo() {
        av_audio_fifo_free(m_audioFifo);
    }

    void addSamples(const AudioFrame& audioFrame);

    void addSamples(AVFrame* frame);

    void addZeros(int padSize);

    AVFrame* popSamples(int frameSize);

    AudioFrame popSamplesAsFrame(size_t numOfPopSamples);

    bool empty() const {
        if (m_audioFifo == nullptr) {
            return true;
        }
        return av_audio_fifo_size(m_audioFifo) == 0;
    }

    int numSamples() const {
        if (m_audioFifo == nullptr) {
            return 0;
        }
        return av_audio_fifo_size(m_audioFifo);
    }

    int sampleRate() const {
        return m_sampleRate;
    }

    void clear() {
        if (m_audioFifo != nullptr) {
            av_audio_fifo_reset(m_audioFifo);
        }
    }

  private:
    AVAudioFifo* getFifo() {
        if (m_audioFifo == nullptr) {
            m_audioFifo = av_audio_fifo_alloc(m_sampleFormat, m_channelLayout.nb_channels, 1);
        }
        return m_audioFifo;
    }

    AVAudioFifo* m_audioFifo = nullptr;
    AVChannelLayout m_channelLayout;
    AVSampleFormat m_sampleFormat = AV_SAMPLE_FMT_NONE;

    int m_sampleRate = 0;
};
}  // namespace p10::media