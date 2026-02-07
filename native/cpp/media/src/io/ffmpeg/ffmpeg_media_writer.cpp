#include "ffmpeg_media_writer.hpp"

#include "ffmpeg_wrap_error.hpp"
#include "ptensor/p10_error.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/time.h>
}

#include "audio_frame.hpp"
#include "video_frame.hpp"

namespace p10::media {

P10Result<std::shared_ptr<FfmpegMediaWriter>>
FfmpegMediaWriter::create(const std::string& path, const MediaParameters& params) {
    AVFormatContext* format_ctx = nullptr;
    // Allocate output format context
    int ret = avformat_alloc_output_context2(&format_ctx, nullptr, nullptr, path.c_str());
    if (ret < 0) {
        return Err(wrap_ffmpeg_error(ret, "Could not allocate output context"));
    }

    // Create video encoder if video parameters are set
    const auto& video_params = params.video_parameters();
    auto video_encoder = std::make_unique<FfmpegVideoEncoder>();
    P10Error error = video_encoder->create(video_params, format_ctx);
    if (error.is_error()) {
        avformat_free_context(format_ctx);
        format_ctx = nullptr;
        return Err(error);
    }

    const auto& audio_params = params.audio_parameters();
    auto audio_encoder = std::make_unique<FfmpegAudioEncoder>();
    error = audio_encoder->create(audio_params, nullptr);
    if (error.is_error()) {
        avformat_free_context(format_ctx);
        format_ctx = nullptr;
        return Err(error);
    }

    // Open output file
    if (!(format_ctx->oformat->flags & AVFMT_NOFILE)) {
        error = wrap_ffmpeg_error(avio_open(&format_ctx->pb, path.c_str(), AVIO_FLAG_WRITE));
        if (error.is_error()) {
            avformat_free_context(format_ctx);
            format_ctx = nullptr;
            return Err(wrap_ffmpeg_error(error, "Could not open output file"));
        }
    }

    // Write header
    error = wrap_ffmpeg_error(avformat_write_header(format_ctx, nullptr));
    if (error.is_error()) {
        if (!(format_ctx->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&format_ctx->pb);
        }
        avformat_free_context(format_ctx);
        format_ctx = nullptr;
        return Err(wrap_ffmpeg_error(error, "Could not write header"));
    }

    auto writer = std::shared_ptr<FfmpegMediaWriter>(new FfmpegMediaWriter(
        format_ctx,
        params,
        std::move(video_encoder),
        std::move(audio_encoder)
    ));
    return Ok(std::move(writer));
}

FfmpegMediaWriter::FfmpegMediaWriter(
    AVFormatContext* format_context,
    const MediaParameters& params,
    std::unique_ptr<FfmpegVideoEncoder> video_encoder,
    std::unique_ptr<FfmpegAudioEncoder> audio_encoder
) :
    format_context_(format_context),
    params_(params),
    video_encoder_(std::move(video_encoder)),
    audio_encoder_(std::move(audio_encoder)) {}

FfmpegMediaWriter::~FfmpegMediaWriter() {
    close();
}

void FfmpegMediaWriter::close() {
    if (closed_) {
        return;
    }
    closed_ = true;

    // Flush encoders
    if (video_encoder_) {
        flush_video_encoder();
    }
    if (audio_encoder_) {
        flush_audio_encoder();
    }

    // Write trailer
    if (format_context_ != nullptr && header_written_) {
        av_write_trailer(format_context_);
    }

    // Close IO
    if (format_context_ != nullptr) {
        if (!(format_context_->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&format_context_->pb);
        }
        avformat_free_context(format_context_);
        format_context_ = nullptr;
    }

    video_encoder_.reset();
    audio_encoder_.reset();
}

MediaParameters FfmpegMediaWriter::get_parameters() const {
    return params_;
}

P10Error FfmpegMediaWriter::write_video(const VideoFrame& frame) {
    if (closed_ || format_context_ == nullptr) {
        return P10Error::InvalidOperation << "Writer is closed";
    }

    if (!video_encoder_) {
        return P10Error::InvalidOperation << "No video encoder configured";
    }

    // Encode the frame
    P10_RETURN_IF_ERROR(video_encoder_->encode_frame(frame, video_pts_));
    video_pts_++;

    // Write all available packets
    while (video_encoder_->has_packets()) {
        AVPacket* pkt = video_encoder_->pop_packet();
        if (pkt != nullptr) {
            P10Error err = write_video_packet(pkt);
            av_packet_free(&pkt);
            if (err.is_error()) {
                return err;
            }
        }
    }

    return P10Error::Ok;
}

P10Error FfmpegMediaWriter::write_audio(const AudioFrame& /*frame*/) {
    if (closed_ || format_context_ == nullptr) {
        return P10Error::InvalidOperation << "Writer is closed";
    }

    if (!audio_encoder_) {
        return P10Error::InvalidOperation << "No audio encoder configured";
    }

    // TODO: Implement audio encoding
    return P10Error::NotImplemented << "Audio writing not yet implemented";
}

P10Error FfmpegMediaWriter::flush_video_encoder() {
    if (!video_encoder_) {
        return P10Error::Ok;
    }

    P10_RETURN_IF_ERROR(video_encoder_->flush());

    while (video_encoder_->has_packets()) {
        AVPacket* pkt = video_encoder_->pop_packet();
        if (pkt != nullptr) {
            P10Error err = write_video_packet(pkt);
            av_packet_free(&pkt);
            if (err.is_error()) {
                return err;
            }
        }
    }

    return P10Error::Ok;
}

P10Error FfmpegMediaWriter::flush_audio_encoder() {
    // TODO: Implement when audio encoder is complete
    return P10Error::Ok;
}

P10Error FfmpegMediaWriter::write_video_packet(AVPacket* packet) {
    if (!video_encoder_ || !video_encoder_->stream()) {
        return P10Error::InvalidOperation << "No video stream";
    }

    // Rescale timestamps to stream time base
    av_packet_rescale_ts(
        packet,
        video_encoder_->codec_context()->time_base,
        video_encoder_->stream()->time_base
    );

    int ret = av_interleaved_write_frame(format_context_, packet);
    if (ret < 0) {
        return wrap_ffmpeg_error(ret, "Failed to write video packet");
    }

    return P10Error::Ok;
}

P10Error FfmpegMediaWriter::write_audio_packet(AVPacket* /*packet*/) {
    // TODO: Implement when audio encoder is complete
    return P10Error::NotImplemented;
}
}  // namespace p10::media
