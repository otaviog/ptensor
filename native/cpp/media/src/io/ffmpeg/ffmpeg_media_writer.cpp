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
    AVFormatContext* format_context_;

     auto video_stream = avformat_new_stream(format_context_, nullptr);
     if (m_audioStream == nullptr) {
        errorOnFail(std::nullopt, "Could not audio add stream");
    }

     auto audio_stream = avformat_new_stream(format_context_, nullptr);
    P10_RETURN_ERR_IF_ERROR(wrap_ffmpeg_error(
        avio_open(&format_context_->pb, path.c_str(), AVIO_FLAG_WRITE),
        "Could not initialize stream parameters"
    ));

    P10_RETURN_ERR_IF_ERROR(wrap_ffmpeg_error(avformat_write_header(format_context_, nullptr)));
}

FfmpegMediaWriter::~FfmpegMediaWriter() override {}

void FfmpegMediaWriter::close() override {}

MediaParameters FfmpegMediaWriter::get_parameters() const override {}

P10Error FfmpegMediaWriter::write_video(const VideoFrame& frame) override {}

P10Error FfmpegMediaWriter::write_audio(const AudioFrame& frame) override {}

}  // namespace p10::media