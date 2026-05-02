#ifndef PTENSOR_MEDIA_H_
#define PTENSOR_MEDIA_H_

#include <stddef.h>
#include <stdint.h>

#include "config.h"
#include "ptensor_error.h"
#include "ptensor_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle for a video frame (RGB24, HxWx3 uint8).
typedef void* P10VideoFrame;

/// Opaque handle for an audio frame (channels x samples float tensor).
typedef void* P10AudioFrame;

/// Opaque handle for a media capture session (reading from file).
typedef void* P10MediaCapture;

/// Opaque handle for a media writer session (writing to file).
typedef void* P10MediaWriter;

/// Rational number used for frame rates and time bases.
typedef struct {
    int64_t num;
    int64_t den;
} P10Rational;

/// Timestamp expressed as a (base, stamp) pair.
/// Seconds = stamp * base.num / base.den
typedef struct {
    P10Rational base;
    int64_t     stamp;
} P10Time;

// ------------------------------------------------------------------ //
// VideoFrame
// ------------------------------------------------------------------ //

/// Allocates a VideoFrame with an RGB24 buffer of the given dimensions.
/// The caller must call p10_video_frame_destroy() when done.
PTENSOR_API P10ErrorEnum p10_video_frame_create(
    P10VideoFrame* frame,
    size_t         width,
    size_t         height
);

/// Destroys a VideoFrame and sets *frame to NULL.
PTENSOR_API P10ErrorEnum p10_video_frame_destroy(P10VideoFrame* frame);

/// Returns the width of the video frame in pixels.
PTENSOR_API size_t p10_video_frame_width(P10VideoFrame frame);

/// Returns the height of the video frame in pixels.
PTENSOR_API size_t p10_video_frame_height(P10VideoFrame frame);

/// Returns the number of channels (always 3 for RGB24).
PTENSOR_API size_t p10_video_frame_channels(P10VideoFrame frame);

/// Returns the presentation timestamp of the frame.
PTENSOR_API P10Time p10_video_frame_time(P10VideoFrame frame);

/// Sets the presentation timestamp of the frame.
PTENSOR_API void p10_video_frame_set_time(P10VideoFrame frame, P10Time time);

/// Returns a non-owning Ptensor view of the frame's image data (HxWxC uint8).
/// The view is valid only while the VideoFrame is alive.
/// The caller must NOT call p10_destroy() on the returned tensor.
PTENSOR_API P10ErrorEnum p10_video_frame_image(P10VideoFrame frame, Ptensor* image_out);

// ------------------------------------------------------------------ //
// AudioFrame
// ------------------------------------------------------------------ //

/// Creates an AudioFrame by copying the data from the given samples tensor.
/// The samples tensor must have shape [channels, num_samples].
/// The caller must call p10_audio_frame_destroy() when done.
PTENSOR_API P10ErrorEnum p10_audio_frame_create(
    P10AudioFrame* frame,
    Ptensor        samples,
    size_t         sample_rate
);

/// Destroys an AudioFrame and sets *frame to NULL.
PTENSOR_API P10ErrorEnum p10_audio_frame_destroy(P10AudioFrame* frame);

/// Returns the total number of samples (per channel).
PTENSOR_API int64_t p10_audio_frame_samples_count(P10AudioFrame frame);

/// Returns the number of audio channels.
PTENSOR_API int64_t p10_audio_frame_channels_count(P10AudioFrame frame);

/// Returns the sample rate in Hz.
PTENSOR_API size_t p10_audio_frame_sample_rate(P10AudioFrame frame);

/// Returns the duration of the frame in seconds.
PTENSOR_API double p10_audio_frame_duration_seconds(P10AudioFrame frame);

/// Returns the presentation timestamp of the audio frame.
PTENSOR_API P10Time p10_audio_frame_time(P10AudioFrame frame);

/// Sets the presentation timestamp of the audio frame.
PTENSOR_API void p10_audio_frame_set_time(P10AudioFrame frame, P10Time time);

/// Returns a non-owning Ptensor view of the frame's samples data.
/// The view is valid only while the AudioFrame is alive.
/// The caller must NOT call p10_destroy() on the returned tensor.
PTENSOR_API P10ErrorEnum p10_audio_frame_samples(P10AudioFrame frame, Ptensor* samples_out);

// ------------------------------------------------------------------ //
// MediaCapture
// ------------------------------------------------------------------ //

/// Opens a media file for reading. Supports any container/codec that FFmpeg
/// can decode. The caller must call p10_media_capture_close() when done.
PTENSOR_API P10ErrorEnum p10_media_capture_open(
    P10MediaCapture* capture,
    const char*      path
);

/// Closes a media capture session and sets *capture to NULL.
PTENSOR_API P10ErrorEnum p10_media_capture_close(P10MediaCapture* capture);

/// Advances to the next frame. Sets *has_frame to 1 if a frame is available,
/// 0 at end-of-stream. Returns a P10ErrorEnum on decode errors.
PTENSOR_API P10ErrorEnum p10_media_capture_next_frame(
    P10MediaCapture capture,
    int*            has_frame
);

/// Decodes the current video frame into *frame_out.
/// p10_media_capture_next_frame() must have been called first.
/// The caller must call p10_video_frame_destroy() on the returned handle.
PTENSOR_API P10ErrorEnum p10_media_capture_get_video(
    P10MediaCapture capture,
    P10VideoFrame*  frame_out
);

/// Decodes the current audio frame into *frame_out.
/// p10_media_capture_next_frame() must have been called first.
/// The caller must call p10_audio_frame_destroy() on the returned handle.
PTENSOR_API P10ErrorEnum p10_media_capture_get_audio(
    P10MediaCapture capture,
    P10AudioFrame*  frame_out
);

/// Returns the video width of the source stream in pixels.
PTENSOR_API int32_t p10_media_capture_video_width(P10MediaCapture capture);

/// Returns the video height of the source stream in pixels.
PTENSOR_API int32_t p10_media_capture_video_height(P10MediaCapture capture);

/// Returns the video frame rate as a rational number.
PTENSOR_API P10Rational p10_media_capture_video_frame_rate(P10MediaCapture capture);

/// Returns the audio sample rate in Hz. Returns 0 if no audio stream.
PTENSOR_API double p10_media_capture_audio_sample_rate(P10MediaCapture capture);

/// Returns the number of audio channels. Returns 0 if no audio stream.
PTENSOR_API size_t p10_media_capture_audio_channels(P10MediaCapture capture);

/// Returns the total number of video frames, or -1 if unknown.
PTENSOR_API int64_t p10_media_capture_video_frame_count(P10MediaCapture capture);

/// Returns the total duration in seconds, or -1.0 if unknown.
PTENSOR_API double p10_media_capture_duration(P10MediaCapture capture);

// ------------------------------------------------------------------ //
// MediaWriter
// ------------------------------------------------------------------ //

/// Opens a media file for writing.
///
/// width / height            - output video dimensions in pixels
/// frame_rate                - output video frame rate (e.g. {1, 25} for 25 fps)
/// audio_sample_rate_hz      - output audio sample rate (e.g. 44100.0); pass 0
///                             to disable audio
/// audio_channels            - number of audio channels; pass 0 to disable audio
///
/// The caller must call p10_media_writer_close() when done.
PTENSOR_API P10ErrorEnum p10_media_writer_open(
    P10MediaWriter* writer,
    const char*     path,
    int32_t         width,
    int32_t         height,
    P10Rational     frame_rate,
    double          audio_sample_rate_hz,
    size_t          audio_channels
);

/// Closes a media writer session and sets *writer to NULL.
PTENSOR_API P10ErrorEnum p10_media_writer_close(P10MediaWriter* writer);

/// Writes one video frame to the output file.
PTENSOR_API P10ErrorEnum p10_media_writer_write_video(
    P10MediaWriter writer,
    P10VideoFrame  frame
);

/// Writes one audio frame to the output file.
PTENSOR_API P10ErrorEnum p10_media_writer_write_audio(
    P10MediaWriter writer,
    P10AudioFrame  frame
);

#ifdef __cplusplus
}
#endif

#endif  // PTENSOR_MEDIA_H_
