#include "ptensor_media.h"

#include <ptensor/media/audio_frame.hpp>
#include <ptensor/media/audio_parameters.hpp>
#include <ptensor/media/io/media_capture.hpp>
#include <ptensor/media/io/media_writer.hpp>
#include <ptensor/media/media_parameters.hpp>
#include <ptensor/media/video_frame.hpp>
#include <ptensor/media/video_parameters.hpp>
#include <ptensor/tensor.hpp>

#include "media_wrapper.hpp"
#include "tensor_wrapper.hpp"
#include "update_error_state.hpp"

using namespace p10::media;

// ------------------------------------------------------------------ //
// VideoFrame
// ------------------------------------------------------------------ //

PTENSOR_API P10ErrorEnum p10_video_frame_create(
    P10VideoFrame* frame,
    size_t         width,
    size_t         height
) {
    auto* vf = new VideoFrame();
    auto err = vf->create(width, height, PixelFormat::RGB24);
    if (err.is_error()) {
        delete vf;
        return p10::update_error_state(err);
    }
    *frame = wrap_video_frame(vf);
    return P10_OK;
}

PTENSOR_API P10ErrorEnum p10_video_frame_destroy(P10VideoFrame* frame) {
    if (frame == nullptr || *frame == nullptr) {
        return P10_OK;
    }
    delete unwrap_video_frame(*frame);
    *frame = nullptr;
    return P10_OK;
}

PTENSOR_API size_t p10_video_frame_width(P10VideoFrame frame) {
    return unwrap_video_frame(frame)->width();
}

PTENSOR_API size_t p10_video_frame_height(P10VideoFrame frame) {
    return unwrap_video_frame(frame)->height();
}

PTENSOR_API size_t p10_video_frame_channels(P10VideoFrame frame) {
    return unwrap_video_frame(frame)->channels();
}

PTENSOR_API P10Time p10_video_frame_time(P10VideoFrame frame) {
    return to_c(unwrap_video_frame(frame)->time());
}

PTENSOR_API void p10_video_frame_set_time(P10VideoFrame frame, P10Time time) {
    unwrap_video_frame(frame)->update_time(from_c(time));
}

PTENSOR_API P10ErrorEnum p10_video_frame_image(P10VideoFrame frame, Ptensor* image_out) {
    auto* vf = unwrap_video_frame(frame);
    p10::Tensor& img = vf->image();

    // Create a non-owning view tensor pointing at the frame's data buffer.
    void* data = img.as_bytes().data();
    auto view = p10::Tensor::from_data(
        data,
        img.shape(),
        p10::TensorOptions().dtype(img.dtype()).stride(img.stride())
    );
    *image_out = p10::wrap(std::move(view));
    return P10_OK;
}

// ------------------------------------------------------------------ //
// AudioFrame
// ------------------------------------------------------------------ //

PTENSOR_API P10ErrorEnum p10_audio_frame_create(
    P10AudioFrame* frame,
    Ptensor        samples,
    size_t         sample_rate
) {
    const p10::Tensor& src = p10::unwrap_ref_const(samples);
    auto clone_result = src.clone();
    if (clone_result.is_error()) {
        return p10::update_error_state(clone_result.unwrap_err());
    }
    *frame = wrap_audio_frame(
        new AudioFrame(std::move(clone_result.unwrap()), sample_rate)
    );
    return P10_OK;
}

PTENSOR_API P10ErrorEnum p10_audio_frame_destroy(P10AudioFrame* frame) {
    if (frame == nullptr || *frame == nullptr) {
        return P10_OK;
    }
    delete unwrap_audio_frame(*frame);
    *frame = nullptr;
    return P10_OK;
}

PTENSOR_API int64_t p10_audio_frame_samples_count(P10AudioFrame frame) {
    return unwrap_audio_frame(frame)->samples_count();
}

PTENSOR_API int64_t p10_audio_frame_channels_count(P10AudioFrame frame) {
    return unwrap_audio_frame(frame)->channels_count();
}

PTENSOR_API size_t p10_audio_frame_sample_rate(P10AudioFrame frame) {
    return unwrap_audio_frame(frame)->sample_rate();
}

PTENSOR_API double p10_audio_frame_duration_seconds(P10AudioFrame frame) {
    return unwrap_audio_frame(frame)->duration_seconds();
}

PTENSOR_API P10Time p10_audio_frame_time(P10AudioFrame frame) {
    return to_c(unwrap_audio_frame(frame)->time());
}

PTENSOR_API void p10_audio_frame_set_time(P10AudioFrame frame, P10Time time) {
    unwrap_audio_frame(frame)->set_time(from_c(time));
}

PTENSOR_API P10ErrorEnum p10_audio_frame_samples(P10AudioFrame frame, Ptensor* samples_out) {
    p10::Tensor& s = unwrap_audio_frame(frame)->samples();

    void* data = s.as_bytes().data();
    auto view = p10::Tensor::from_data(
        data,
        s.shape(),
        p10::TensorOptions().dtype(s.dtype()).stride(s.stride())
    );
    *samples_out = p10::wrap(std::move(view));
    return P10_OK;
}

// ------------------------------------------------------------------ //
// MediaCapture
// ------------------------------------------------------------------ //

PTENSOR_API P10ErrorEnum p10_media_capture_open(
    P10MediaCapture* capture,
    const char*      path
) {
    auto result = MediaCapture::open_file(path);
    if (result.is_error()) {
        return p10::update_error_state(result.unwrap_err());
    }
    *capture = wrap_capture(new MediaCapture(std::move(result.unwrap())));
    return P10_OK;
}

PTENSOR_API P10ErrorEnum p10_media_capture_close(P10MediaCapture* capture) {
    if (capture == nullptr || *capture == nullptr) {
        return P10_OK;
    }
    auto* c = unwrap_capture(*capture);
    c->close();
    delete c;
    *capture = nullptr;
    return P10_OK;
}

PTENSOR_API P10ErrorEnum p10_media_capture_next_frame(
    P10MediaCapture capture,
    int*            has_frame
) {
    auto result = unwrap_capture(capture)->next_frame();
    if (result.is_error()) {
        return p10::update_error_state(result.unwrap_err());
    }
    *has_frame = result.unwrap() ? 1 : 0;
    return P10_OK;
}

PTENSOR_API P10ErrorEnum p10_media_capture_get_video(
    P10MediaCapture capture,
    P10VideoFrame*  frame_out
) {
    auto* vf = new VideoFrame();
    auto err = unwrap_capture(capture)->get_video(*vf);
    if (err.is_error()) {
        delete vf;
        return p10::update_error_state(err);
    }
    *frame_out = wrap_video_frame(vf);
    return P10_OK;
}

PTENSOR_API P10ErrorEnum p10_media_capture_get_audio(
    P10MediaCapture capture,
    P10AudioFrame*  frame_out
) {
    auto* af = new AudioFrame();
    auto err = unwrap_capture(capture)->get_audio(*af);
    if (err.is_error()) {
        delete af;
        return p10::update_error_state(err);
    }
    *frame_out = wrap_audio_frame(af);
    return P10_OK;
}

PTENSOR_API int32_t p10_media_capture_video_width(P10MediaCapture capture) {
    return unwrap_capture(capture)->get_parameters().video_parameters().width();
}

PTENSOR_API int32_t p10_media_capture_video_height(P10MediaCapture capture) {
    return unwrap_capture(capture)->get_parameters().video_parameters().height();
}

PTENSOR_API P10Rational p10_media_capture_video_frame_rate(P10MediaCapture capture) {
    return to_c(
        unwrap_capture(capture)->get_parameters().video_parameters().frame_rate()
    );
}

PTENSOR_API double p10_media_capture_audio_sample_rate(P10MediaCapture capture) {
    return unwrap_capture(capture)->get_parameters().audio_parameters().audio_sample_rate_hz();
}

PTENSOR_API size_t p10_media_capture_audio_channels(P10MediaCapture capture) {
    return unwrap_capture(capture)->get_parameters().audio_parameters().audio_channels();
}

PTENSOR_API int64_t p10_media_capture_video_frame_count(P10MediaCapture capture) {
    auto opt = unwrap_capture(capture)->video_frame_count();
    return opt.has_value() ? *opt : -1;
}

PTENSOR_API double p10_media_capture_duration(P10MediaCapture capture) {
    auto opt = unwrap_capture(capture)->duration();
    return opt.has_value() ? *opt : -1.0;
}

// ------------------------------------------------------------------ //
// MediaWriter
// ------------------------------------------------------------------ //

PTENSOR_API P10ErrorEnum p10_media_writer_open(
    P10MediaWriter* writer,
    const char*     path,
    int32_t         width,
    int32_t         height,
    P10Rational     frame_rate,
    double          audio_sample_rate_hz,
    size_t          audio_channels
) {
    MediaParameters params;
    params.video_parameters()
        .width(width)
        .height(height)
        .frame_rate(from_c(frame_rate));

    if (audio_sample_rate_hz > 0.0 && audio_channels > 0) {
        params.audio_parameters()
            .audio_sample_rate_hz(audio_sample_rate_hz)
            .audio_channels(audio_channels);
    }

    auto result = MediaWriter::open_file(path, params);
    if (result.is_error()) {
        return p10::update_error_state(result.unwrap_err());
    }
    *writer = wrap_writer(new MediaWriter(std::move(result.unwrap())));
    return P10_OK;
}

PTENSOR_API P10ErrorEnum p10_media_writer_close(P10MediaWriter* writer) {
    if (writer == nullptr || *writer == nullptr) {
        return P10_OK;
    }
    auto* w = unwrap_writer(*writer);
    w->close();
    delete w;
    *writer = nullptr;
    return P10_OK;
}

PTENSOR_API P10ErrorEnum p10_media_writer_write_video(
    P10MediaWriter writer,
    P10VideoFrame  frame
) {
    auto err = unwrap_writer(writer)->write_video(*unwrap_video_frame(frame));
    if (err.is_error()) {
        return p10::update_error_state(err);
    }
    return P10_OK;
}

PTENSOR_API P10ErrorEnum p10_media_writer_write_audio(
    P10MediaWriter writer,
    P10AudioFrame  frame
) {
    auto err = unwrap_writer(writer)->write_audio(*unwrap_audio_frame(frame));
    if (err.is_error()) {
        return p10::update_error_state(err);
    }
    return P10_OK;
}
