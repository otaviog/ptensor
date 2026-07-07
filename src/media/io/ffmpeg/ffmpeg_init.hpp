#pragma once

namespace p10::media {

/// Route libav* log output to `<get_log_directory()>/ffmpeg.log`.
///
/// Idempotent and thread-safe: the first call installs the FFmpeg log callback,
/// later calls are no-ops. Call it at every entry point that touches libav*
/// (capture, writer, device enumeration). If the log file cannot be opened,
/// FFmpeg keeps its default stderr logging.
void ffmpeg_init();

}  // namespace p10::media
