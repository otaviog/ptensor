
#include "ffmpeg_file_media_capture.hpp"

#include <memory>
#include <mutex>

#include "ffmpeg_audio_decoder.hpp"
#include "ffmpeg_memory.hpp"
#include "ffmpeg_video_decoder.hpp"
#include "ffmpeg_wrap_error.hpp"
#include "media_parameters.hpp"

extern "C" {
#include <libavcodec/packet.h>
#include <libavformat/avformat.h>
}

namespace p10::media {

FfmpegFileMediaCapture::~FfmpegFileMediaCapture() {
    close();
}

P10Result<std::shared_ptr<FfmpegFileMediaCapture>>
FfmpegFileMediaCapture::open(const std::string& path) {
    AVFormatContext* format_ctx = nullptr;
    avformat_open_input(&format_ctx, path.c_str(), nullptr, nullptr);

    if (!format_ctx) {
        return Err(P10Error(P10Error::IoError, "Failed to open media file: " + path));
    }

    avformat_find_stream_info(format_ctx, nullptr);

    const int video_stream_idx =
        av_find_best_stream(format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    std::shared_ptr<FfmpegVideoDecoder> video_decoder;
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

    const int audio_stream_idx =
        av_find_best_stream(format_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    std::shared_ptr<FfmpegAudioDecoder> audio_decoder;
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

    capture->start_decoding_thread();
    return Ok(std::move(capture));
}

void FfmpegFileMediaCapture::close() {
    status_ = CaptureStatus::Stopped;

    video_queue_.cancel();

    if (decode_thread_.joinable()) {
        decode_thread_.join();
    }
    avformat_close_input(&format_ctx_);
    format_ctx_ = nullptr;
}

MediaParameters FfmpegFileMediaCapture::get_parameters() const {
    if (!is_open()) {
        return MediaParameters();
    }

    MediaParameters params;
    params.video_parameters(video_decoder_->get_video_parameters());
    // TODO: Get audio parameters
    // params.audio_parameters(audio_decoder_->get_audio_parameters());
    return params;
}

P10Result<bool> FfmpegFileMediaCapture::next_frame() {
    CaptureStatus capture_status = status_;
    if (capture_status == CaptureStatus::Reading) {
        current_frame_ = video_queue_.wait_and_pop();
        if (current_frame_.has_value()) {
            return Ok(true);
        }
        return Ok(false);
    } else if (capture_status == CaptureStatus::EndOfFile) {
        current_frame_ = video_queue_.try_pop();
        if (current_frame_.has_value()) {
            return Ok(true);
        } else {
            return Ok(false);
        }
    } else if (capture_status == CaptureStatus::Stopped) {
        return Ok(false);
    } else if (capture_status == CaptureStatus::Error) {
        return Err(last_error_);
    } else {
        return Err(P10Error::NotImplemented << "Invalid capture status");
    }
}

P10Error FfmpegFileMediaCapture::get_video(VideoFrame& frame) {
    if (current_frame_.has_value()) {
        frame = std::move(current_frame_.value());
        return P10Error::Ok;
    } else {
        return P10Error::InvalidArgument << "No video frame available, did you call next_frame()?";
    }
}

P10Error FfmpegFileMediaCapture::get_audio(AudioFrame& /*frame*/) {
    return P10Error::NotImplemented;
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
    UniqueAvPacketRef pkt(av_packet_alloc());

    const int read_ret_code = av_read_frame(format_ctx_, pkt.get());

    if (read_ret_code < 0) {
        if (read_ret_code == AVERROR_EOF) {
            status_ = CaptureStatus::EndOfFile;
        } else {
            status_ = CaptureStatus::Error;
            last_error_ = wrap_ffmpeg_error(read_ret_code, "Failed to read frame");
        }
        return;
    }

    if (video_decoder_ != nullptr && pkt->stream_index == video_decoder_->index()) {
        decode_video_packet(pkt.get());
    } else if (audio_decoder_ != nullptr && pkt->stream_index == audio_decoder_->index()) {
        decode_audio_packet(pkt.get());
    }
}

void FfmpegFileMediaCapture::decode_video_packet(const AVPacket* pkt) {
    auto decode_status = video_decoder_->decode_packet(pkt, video_queue_);
    if (decode_status.is_error()) {
        status_ = CaptureStatus::Error;
        last_error_ = decode_status.error();
        return;
    }

    switch (decode_status.unwrap()) {
        case FfmpegVideoDecoder::DecodeStatus::Eof:
            status_ = CaptureStatus::EndOfFile;
            break;
        case FfmpegVideoDecoder::DecodeStatus::Cancelled:
            status_ = CaptureStatus::Stopped;
            break;
        case FfmpegVideoDecoder::DecodeStatus::FrameDecoded:
        case FfmpegVideoDecoder::DecodeStatus::Again:
        default:
            // Do nothing, continue reading packets
            break;
    }
}

void FfmpegFileMediaCapture::decode_audio_packet(const AVPacket*) {
    //AVFrame* frame = audio_decoder_->decode_packet(pkt);
}
}  // namespace p10::media