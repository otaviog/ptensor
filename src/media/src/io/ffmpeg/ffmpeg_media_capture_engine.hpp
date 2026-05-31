#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <thread>
#include <utility>

#include "../media_capture.impl.hpp"
#include "video_frame.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

#include "../video_queue.hpp"

namespace p10::media {
class FfmpegVideoDecoder;
class FfmpegAudioDecoder;

/// Shared decode engine for FFmpeg-backed captures.
///
/// Owns the demuxer (AVFormatContext), the audio/video decoders, the frame
/// queue and the background read/decode thread. It is agnostic to where the
/// media comes from: file-based and live device captures both build on it and
/// only differ in how the AVFormatContext is opened and in their lifecycle
/// reporting (duration / frame count). The engine takes ownership of an
/// already-opened format context plus the decoders discovered from it.
class FfmpegMediaCaptureEngine: public MediaCapture::Impl {
  public:
    enum CaptureStatus : int8_t { Reading, Stopped, Error, EndOfFile };

    FfmpegMediaCaptureEngine(const FfmpegMediaCaptureEngine&) = delete;
    FfmpegMediaCaptureEngine& operator=(const FfmpegMediaCaptureEngine&) = delete;
    FfmpegMediaCaptureEngine(FfmpegMediaCaptureEngine&&) = delete;
    FfmpegMediaCaptureEngine& operator=(FfmpegMediaCaptureEngine&&) = delete;

    ~FfmpegMediaCaptureEngine() override;

    void close() override;

    MediaParameters get_parameters() const override;

    P10Result<MediaCapture::NextFrameResult> next_frame(MediaCapture::WaitMode wait) override;

    P10Error get_video(VideoFrame& frame) override;

    P10Error get_audio(AudioFrame& frame) override;

    bool is_open() const {
        return format_ctx_ != nullptr;
    }

  protected:
    FfmpegMediaCaptureEngine(
        AVFormatContext* format_ctx,
        std::shared_ptr<FfmpegAudioDecoder> audio_decoder,
        std::shared_ptr<FfmpegVideoDecoder> video_decoder
    ) :
        format_ctx_(format_ctx),
        audio_decoder_(std::move(audio_decoder)),
        video_decoder_(std::move(video_decoder)) {}

    /// Open the demuxer for `url` using the given input format (nullptr for
    /// auto-detected files) and demuxer options, then build the decoders. On
    /// success the decoders are returned to the caller, which wraps them in the
    /// concrete engine subclass. `options` is consumed (freed) by this call.
    struct OpenResult {
        AVFormatContext* format_ctx = nullptr;
        std::shared_ptr<FfmpegAudioDecoder> audio_decoder;
        std::shared_ptr<FfmpegVideoDecoder> video_decoder;
    };

    static P10Result<OpenResult>
    open_format(const std::string& url, const AVInputFormat* fmt, AVDictionary** options);

    void start_decoding_thread();

    /// Stop the decode thread, seek the demuxer to `seconds`, flush codec
    /// buffers and restart decoding. Only valid for seekable sources (files).
    P10Error seek_to(double seconds);

    const FfmpegVideoDecoder* video_decoder() const {
        return video_decoder_.get();
    }

    const AVFormatContext* format_ctx() const {
        return format_ctx_;
    }

  private:
    void read_packets_loop();
    void read_next_packet();
    void decode_video_packet(const AVPacket* pkt);
    void decode_audio_packet(const AVPacket* pkt);

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
