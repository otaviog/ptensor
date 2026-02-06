#include "ffmpeg/ffmpeg_media_writer.hpp"

namespace p10::media {

P10Result<MediaWriter>
MediaWriter::open_file(const std::string& path, const MediaParameters& params) {
    auto result = FfmpegMediaWriter::create(path, params);
    if (result.is_error()) {
        return Err(result.error());
    }
    return Ok(MediaWriter(result.unwrap()));
}

void MediaWriter::close() {
    impl_->close();
}

MediaParameters MediaWriter::get_parameters() const {
    return impl_->get_parameters();
}

P10Error MediaWriter::write_video(const VideoFrame& frame) {
    return impl_->write_video(frame);
}

P10Error MediaWriter::write_audio(const AudioFrame& frame) {
    return impl_->write_audio(frame);
}

}  // namespace p10::media
