#include "ffmpeg_media_capture_engine.hpp"

#include <cassert>
#include <memory>

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

FfmpegMediaCaptureEngine::~FfmpegMediaCaptureEngine() {
    close();
}

P10Result<FfmpegMediaCaptureEngine::OpenResult> FfmpegMediaCaptureEngine::open_format(
    const std::string& url,
    const AVInputFormat* fmt,
    AVDictionary** options
) {
    AVFormatContext* format_ctx = nullptr;
    const P10Error open_error =
        wrap_ffmpeg_error(avformat_open_input(&format_ctx, url.c_str(), fmt, options));
    if (open_error.is_error()) {
        return Err(open_error);
    }
    assert(format_ctx != nullptr);
    P10_RETURN_ERR_IF_ERROR(wrap_ffmpeg_error(avformat_find_stream_info(format_ctx, nullptr)));

    OpenResult result;
    result.format_ctx = format_ctx;

    const int video_stream_idx =
        av_find_best_stream(format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_idx >= 0) {
        AVStream* video_stream = format_ctx->streams[video_stream_idx];
        const AVCodec* video_codec = avcodec_find_decoder(video_stream->codecpar->codec_id);

        AVCodecContext* video_codec_ctx = avcodec_alloc_context3(video_codec);
        P10_RETURN_ERR_IF_ERROR(wrap_ffmpeg_error(
            avcodec_parameters_to_context(video_codec_ctx, video_stream->codecpar)
        ));
        P10_RETURN_ERR_IF_ERROR(
            wrap_ffmpeg_error(avcodec_open2(video_codec_ctx, video_codec, nullptr))
        );

        result.video_decoder =
            std::make_shared<FfmpegVideoDecoder>(video_stream, video_codec_ctx, video_stream_idx);
    }

    const int audio_stream_idx =
        av_find_best_stream(format_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (audio_stream_idx >= 0) {
        AVStream const* audio_stream = format_ctx->streams[audio_stream_idx];
        const AVCodec* audio_codec = avcodec_find_decoder(audio_stream->codecpar->codec_id);

        AVCodecContext* audio_codec_ctx = avcodec_alloc_context3(audio_codec);
        P10_RETURN_ERR_IF_ERROR(wrap_ffmpeg_error(
            avcodec_parameters_to_context(audio_codec_ctx, audio_stream->codecpar)
        ));
        P10_RETURN_ERR_IF_ERROR(
            wrap_ffmpeg_error(avcodec_open2(audio_codec_ctx, audio_codec, nullptr))
        );

        result.audio_decoder =
            std::make_shared<FfmpegAudioDecoder>(audio_codec_ctx, audio_stream_idx);
    }

    return Ok(std::move(result));
}

void FfmpegMediaCaptureEngine::close() {
    status_ = CaptureStatus::Stopped;

    video_queue_.cancel();

    if (decode_thread_.joinable()) {
        decode_thread_.join();
    }
    avformat_close_input(&format_ctx_);
    format_ctx_ = nullptr;
}

MediaParameters FfmpegMediaCaptureEngine::get_parameters() const {
    if (!is_open()) {
        return MediaParameters();
    }

    MediaParameters params;
    if (video_decoder_ != nullptr) {
        params.video_parameters(video_decoder_->get_video_parameters());
    }
    // TODO: Get audio parameters
    // params.audio_parameters(audio_decoder_->get_audio_parameters());
    return params;
}

P10Result<MediaCapture::NextFrameResult>
FfmpegMediaCaptureEngine::next_frame(MediaCapture::WaitMode wait) {
    const CaptureStatus capture_status = status_;
    if (capture_status == CaptureStatus::Reading) {
        if (wait == MediaCapture::WaitMode::Block) {
            current_frame_ = video_queue_.wait_and_pop();
        } else {
            current_frame_ = video_queue_.try_pop();
        }
        return Ok(current_frame_.has_value() ? MediaCapture::Available : MediaCapture::NotReady);
    }
    if (capture_status == CaptureStatus::EndOfFile) {
        current_frame_ = video_queue_.try_pop();
        return Ok(current_frame_.has_value() ? MediaCapture::Available : MediaCapture::Done);
    }
    if (capture_status == CaptureStatus::Stopped) {
        return Ok(MediaCapture::Done);
    }
    if (capture_status == CaptureStatus::Error) {
        return Err(last_error_);
    }
    return Err(P10Error::NotImplemented << "Invalid capture status");
}

P10Error FfmpegMediaCaptureEngine::get_video(VideoFrame& frame) {
    if (current_frame_.has_value()) {
        frame = std::move(current_frame_.value());
        return P10Error::Ok;
    }
    return P10Error::InvalidArgument << "No video frame available, did you call next_frame()?";
}

P10Error FfmpegMediaCaptureEngine::get_audio(AudioFrame& /*frame*/) {
    return P10Error::NotImplemented;
}

void FfmpegMediaCaptureEngine::start_decoding_thread() {
    if (status_ == CaptureStatus::Stopped && is_open()) {
        if (decode_thread_.joinable()) {
            decode_thread_.join();
        }
        status_ = CaptureStatus::Reading;
        decode_thread_ = std::thread(&FfmpegMediaCaptureEngine::read_packets_loop, this);
    }
}

void FfmpegMediaCaptureEngine::read_packets_loop() {
    while (status_ == CaptureStatus::Reading) {
        read_next_packet();
    }
}

void FfmpegMediaCaptureEngine::read_next_packet() {
    UniqueAvPacketRef pkt(av_packet_alloc());

    const int read_ret_code = av_read_frame(format_ctx_, pkt.get());

    if (read_ret_code < 0) {
        if (read_ret_code == AVERROR(EAGAIN)) {
            return;  // device not ready yet; retry next iteration
        }
        if (read_ret_code == AVERROR_EOF) {
            status_ = CaptureStatus::EndOfFile;
            if (video_decoder_ != nullptr) {
                decode_video_packet(nullptr);
            }
        } else {
            status_ = CaptureStatus::Error;
            last_error_ = wrap_ffmpeg_error(read_ret_code, "Failed to read frame");
        }
        video_queue_.cancel();
        return;
    }

    if (video_decoder_ != nullptr && pkt->stream_index == video_decoder_->index()) {
        decode_video_packet(pkt.get());
    } else if (audio_decoder_ != nullptr && pkt->stream_index == audio_decoder_->index()) {
        decode_audio_packet(pkt.get());
    }
}

void FfmpegMediaCaptureEngine::decode_video_packet(const AVPacket* pkt) {
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

void FfmpegMediaCaptureEngine::decode_audio_packet(const AVPacket*) {
    //AVFrame* frame = audio_decoder_->decode_packet(pkt);
}

P10Error FfmpegMediaCaptureEngine::seek_to(double seconds) {
    status_ = CaptureStatus::Stopped;
    video_queue_.cancel();
    if (decode_thread_.joinable()) {
        decode_thread_.join();
    }

    auto ts = static_cast<int64_t>(seconds * AV_TIME_BASE);
    const int ret = avformat_seek_file(format_ctx_, -1, INT64_MIN, ts, INT64_MAX, 0);
    if (ret < 0) {
        return wrap_ffmpeg_error(ret, "Seek failed");
    }

    if (video_decoder_ != nullptr) {
        video_decoder_->flush();
    }
    if (audio_decoder_ != nullptr) {
        audio_decoder_->flush();
    }

    video_queue_.flush();
    start_decoding_thread();
    return P10Error::Ok;
}

}  // namespace p10::media
