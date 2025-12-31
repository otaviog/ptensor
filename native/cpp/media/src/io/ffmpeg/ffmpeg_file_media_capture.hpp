#pragma once

#include <atomic>
#include <memory>
#include <thread>

#include "../media_capture.impl.hpp"
#include "ptensor/p10_error.hpp"
#include "ptensor/p10_result.hpp"
#include "video_frame.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

#include "../video_queue.hpp"

namespace p10::media {
class FfmpegVideoDecoder;
class FfmpegAudioDecoder;

class FfmpegFileMediaCapture: public MediaCapture::Impl {
  public:
    enum CaptureStatus : int8_t { Reading, Stopped, Error, EndOfFile };

    ~FfmpegFileMediaCapture() override;

    static P10Result<std::shared_ptr<FfmpegFileMediaCapture>> open(const std::string& path);

    void close() override;

    MediaParameters get_parameters() const override;

    P10Result<bool> next_frame() override;

    P10Error get_video(VideoFrame& frame) override;

    P10Error get_audio(AudioFrame& frame) override;

    bool is_open() const {
        return format_ctx_ != nullptr;
    }

  private:
    FfmpegFileMediaCapture(
        AVFormatContext* format_ctx,
        std::shared_ptr<FfmpegAudioDecoder> audio_decoder,
        std::shared_ptr<FfmpegVideoDecoder> video_decoder
    ) :
        format_ctx_(format_ctx),
        audio_decoder_(audio_decoder),
        video_decoder_(video_decoder) {}

    void read_packets_loop();
    void read_next_packet();
    void decode_video_packet(const AVPacket* pkt);
    void decode_audio_packet(const AVPacket* pkt);
    void start_decoding_thread();

    std::atomic<CaptureStatus> status_ = CaptureStatus::Stopped;
    
    AVFormatContext* format_ctx_ = nullptr;

    std::thread decode_thread_;


    std::shared_ptr<FfmpegAudioDecoder> audio_decoder_;
    std::shared_ptr<FfmpegVideoDecoder> video_decoder_;

    VideoQueue video_queue_ {30};
    std::optional<VideoFrame> current_frame_;
    P10Error last_error_;
};

}  // namespace p10::media