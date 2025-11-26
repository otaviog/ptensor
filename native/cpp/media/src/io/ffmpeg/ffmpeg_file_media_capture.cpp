
#include "ffmpeg_file_media_capture.hpp"

#include <memory>
#include <mutex>

#include "ffmpeg_audio_decoder.hpp"
#include "ffmpeg_video_decoder.hpp"
#include "ffmpeg_wrap_error.hpp"
#include "io/media_parameters.hpp"

extern "C" {
#include <libavcodec/packet.h>
#include <libavformat/avformat.h>
}

namespace p10::media {

P10Result<std::shared_ptr<FfmpegFileMediaCapture>>
FfmpegFileMediaCapture::open(const std::string& path) {
    AVFormatContext* format_ctx = nullptr;
    avformat_open_input(&format_ctx, path.c_str(), nullptr, nullptr);

    if (!format_ctx) {
        return Err(P10Error(P10Error::IoError, "Failed to open media file: " + path));
    }

    avformat_find_stream_info(format_ctx, nullptr);
    int video_stream_idx = av_find_best_stream(format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    int audio_stream_idx = av_find_best_stream(format_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);

    std::shared_ptr<FfmpegVideoDecoder> video_decoder;
    std::shared_ptr<FfmpegAudioDecoder> audio_decoder;
    if (video_stream_idx >= 0) {
        AVStream* video_stream = format_ctx->streams[video_stream_idx];
        const AVCodec* video_codec = avcodec_find_decoder(video_stream->codecpar->codec_id);

        AVCodecContext* video_codec_ctx = avcodec_alloc_context3(video_codec);
        avcodec_parameters_to_context(video_codec_ctx, video_stream->codecpar);
        avcodec_open2(video_codec_ctx, video_codec, nullptr);

        // Create decoder
        video_decoder =
            std::make_shared<FfmpegVideoDecoder>(video_stream, video_codec_ctx, video_stream_idx);
    }

    if (audio_stream_idx >= 0) {
        AVStream* audio_stream = format_ctx->streams[audio_stream_idx];
        const AVCodec* audio_codec = avcodec_find_decoder(audio_stream->codecpar->codec_id);

        AVCodecContext* audio_codec_ctx = avcodec_alloc_context3(audio_codec);
        avcodec_parameters_to_context(audio_codec_ctx, audio_stream->codecpar);
        avcodec_open2(audio_codec_ctx, audio_codec, nullptr);

        audio_decoder =
            std::make_shared<FfmpegAudioDecoder>(audio_stream, audio_codec_ctx, audio_stream_idx);
    }
    auto capture = std::shared_ptr<FfmpegFileMediaCapture>(
        new FfmpegFileMediaCapture(format_ctx, audio_decoder, video_decoder)
    );
    return Ok(std::move(capture));
}

MediaParameters FfmpegFileMediaCapture::get_parameters() const {
    return MediaParameters();
}

P10Error FfmpegFileMediaCapture::next_frame() {
    current_frame_ = video_queue_.wait_and_pop();
    return P10Error::Ok;
}

P10Result<VideoFrame> FfmpegFileMediaCapture::get_video() {
    if (current_frame_.has_value()) {
        return Ok(std::move(current_frame_.value()));
    } else {
        return Err(
            P10Error::InvalidArgument << "No video frame available, did you call next_frame()?"
        );
    }
}

P10Result<AudioFrame> FfmpegFileMediaCapture::get_audio() {
    return Err(P10Error::NotImplemented);
}

void FfmpegFileMediaCapture::start_decoding_thread() {
    if (status_ == CaptureStatus::Stopped && is_open()) {
        if (decode_thread_.joinable()) {
            decode_thread_.join();
        }
        decode_thread_ = std::thread(&FfmpegFileMediaCapture::read_packets_loop, this);
    }
}

void FfmpegFileMediaCapture::read_packets_loop() {
    status_ = CaptureStatus::Reading;
    while (status_ == CaptureStatus::Reading) {
        read_next_packet();
    }
}

void FfmpegFileMediaCapture::read_next_packet() {
    std::unique_lock lock(mutex_);

    UniqueAvPacketRef pkt(av_packet_alloc());

    int read_ret_code = av_read_frame(format_ctx_, pkt.get());

    if (read_ret_code < 0) {
        if (read_ret_code == AVERROR_EOF) {
            status_ = CaptureStatus::EndOfFile;
        } else {
            status_ = CaptureStatus::Error;
            last_error_ = wrap_error(read_ret_code, "Failed to read frame");
        }
        return;
    }

    if (pkt->stream_index == video_decoder_->index()) {
        decode_video_packet(pkt.get());
    } else if (pkt->stream_index == audio_decoder_->index()) {
        decode_audio_packet(pkt.get());
    }
}

void FfmpegFileMediaCapture::decode_video_packet(const AVPacket* pkt) {
    VideoFrame current_video_frame_;
    video_decoder_->decode_packet(pkt, current_video_frame_);
}

void FfmpegFileMediaCapture::decode_audio_packet(const AVPacket*) {
    //AVFrame* frame = audio_decoder_->decode_packet(pkt);
}
}  // namespace p10::media