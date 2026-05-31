#pragma once

#include <ptensor/media/audio_frame.hpp>
#include <ptensor/media/io/media_capture.hpp>
#include <ptensor/media/io/media_writer.hpp>
#include <ptensor/media/video_frame.hpp>

#include "ptensor_media.h"

namespace p10::media {

// ------------------------------------------------------------------ //
// VideoFrame
// ------------------------------------------------------------------ //

inline VideoFrame* unwrap_video_frame(P10VideoFrame h) {
    assert(h != nullptr && "Null P10VideoFrame handle");
    return reinterpret_cast<VideoFrame*>(h);
}

inline P10VideoFrame wrap_video_frame(VideoFrame* p) {
    return reinterpret_cast<P10VideoFrame>(p);
}

// ------------------------------------------------------------------ //
// AudioFrame
// ------------------------------------------------------------------ //

inline AudioFrame* unwrap_audio_frame(P10AudioFrame h) {
    assert(h != nullptr && "Null P10AudioFrame handle");
    return reinterpret_cast<AudioFrame*>(h);
}

inline P10AudioFrame wrap_audio_frame(AudioFrame* p) {
    return reinterpret_cast<P10AudioFrame>(p);
}

// ------------------------------------------------------------------ //
// MediaCapture
// ------------------------------------------------------------------ //

inline MediaCapture* unwrap_capture(P10MediaCapture h) {
    assert(h != nullptr && "Null P10MediaCapture handle");
    return reinterpret_cast<MediaCapture*>(h);
}

inline P10MediaCapture wrap_capture(MediaCapture* p) {
    return reinterpret_cast<P10MediaCapture>(p);
}

// ------------------------------------------------------------------ //
// MediaWriter
// ------------------------------------------------------------------ //

inline MediaWriter* unwrap_writer(P10MediaWriter h) {
    assert(h != nullptr && "Null P10MediaWriter handle");
    return reinterpret_cast<MediaWriter*>(h);
}

inline P10MediaWriter wrap_writer(MediaWriter* p) {
    return reinterpret_cast<P10MediaWriter>(p);
}

// ------------------------------------------------------------------ //
// Time / Rational conversion helpers
// ------------------------------------------------------------------ //

inline P10Rational to_c(const Rational& r) {
    return P10Rational {r.num(), r.den()};
}

inline Rational from_c(P10Rational r) {
    return Rational {r.num, r.den};
}

inline P10Time to_c(const Time& t) {
    return P10Time {to_c(t.base()), t.stamp()};
}

inline Time from_c(P10Time t) {
    return Time {from_c(t.base), t.stamp};
}

}  // namespace p10::media
