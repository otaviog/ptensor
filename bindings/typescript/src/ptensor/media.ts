import { ffiInt, ffiU64, newHandleBuf, readHandle } from './_internal';
import {
  p10_audio_frame_channels_count,
  p10_audio_frame_create,
  p10_audio_frame_destroy,
  p10_audio_frame_duration_seconds,
  p10_audio_frame_sample_rate,
  p10_audio_frame_samples,
  p10_audio_frame_samples_count,
  p10_audio_frame_set_time_parts,
  p10_audio_frame_time_base_den,
  p10_audio_frame_time_base_num,
  p10_audio_frame_time_stamp,
  p10_media_capture_audio_channels,
  p10_media_capture_audio_sample_rate,
  p10_media_capture_close,
  p10_media_capture_duration,
  p10_media_capture_get_audio,
  p10_media_capture_get_video,
  p10_media_capture_next_frame,
  p10_media_capture_open,
  p10_media_capture_video_frame_count,
  p10_media_capture_video_frame_rate_den,
  p10_media_capture_video_frame_rate_num,
  p10_media_capture_video_height,
  p10_media_capture_video_width,
  p10_media_writer_close,
  p10_media_writer_open_ffi,
  p10_media_writer_write_audio,
  p10_media_writer_write_video,
  p10_video_frame_channels,
  p10_video_frame_create,
  p10_video_frame_destroy,
  p10_video_frame_height,
  p10_video_frame_image,
  p10_video_frame_set_time_parts,
  p10_video_frame_time_base_den,
  p10_video_frame_time_base_num,
  p10_video_frame_time_stamp,
  p10_video_frame_width,
} from './backends/bun/ffi';
import { P10Error } from './p10Error';
import { _getRawHandle, _wrapHandle, type Tensor } from './tensor';

// ------------------------------------------------------------------ //
// Value types
// ------------------------------------------------------------------ //

/** Rational number (e.g. frame rate {num: 25n, den: 1n} = 25 fps). */
export interface Rational {
  num: bigint;
  den: bigint;
}

/** Presentation timestamp. Seconds = stamp * base.num / base.den. */
export interface MediaTime {
  base: Rational;
  stamp: bigint;
}

// ------------------------------------------------------------------ //
// VideoFrame
// ------------------------------------------------------------------ //

export interface VideoFrame {
  /** Frame width in pixels. */
  getWidth(): number;
  /** Frame height in pixels. */
  getHeight(): number;
  /** Number of channels (always 3 for RGB24). */
  getChannels(): number;
  /** Presentation timestamp. */
  getTime(): MediaTime;
  /** Set the presentation timestamp. */
  setTime(t: MediaTime): void;
  /**
   * Returns a non-owning Tensor view of the frame image data (H×W×C uint8).
   * The view is valid only while this VideoFrame is alive.
   * Do NOT call delete() on the returned Tensor.
   */
  getImage(): Tensor;
  /** Releases the native frame handle. */
  delete(): void;
}

class VideoFrameImpl implements VideoFrame {
  readonly _buf: BigUint64Array;

  constructor(buf: BigUint64Array) {
    this._buf = buf;
  }

  getWidth(): number {
    return Number(ffiU64(p10_video_frame_width(readHandle(this._buf))));
  }
  getHeight(): number {
    return Number(ffiU64(p10_video_frame_height(readHandle(this._buf))));
  }
  getChannels(): number {
    return Number(ffiU64(p10_video_frame_channels(readHandle(this._buf))));
  }

  getTime(): MediaTime {
    const h = readHandle(this._buf);
    return {
      base: {
        num: ffiU64(p10_video_frame_time_base_num(h)),
        den: ffiU64(p10_video_frame_time_base_den(h)),
      },
      stamp: ffiU64(p10_video_frame_time_stamp(h)),
    };
  }

  setTime(t: MediaTime): void {
    p10_video_frame_set_time_parts(readHandle(this._buf), t.base.num, t.base.den, t.stamp);
  }

  getImage(): Tensor {
    const imgBuf = newHandleBuf();
    P10Error.check(ffiInt(p10_video_frame_image(readHandle(this._buf), imgBuf)));
    return _wrapHandle(imgBuf);
  }

  delete(): void {
    if (this._buf[0] !== 0n) {
      P10Error.check(ffiInt(p10_video_frame_destroy(this._buf)));
    }
  }
}

/** Creates a new blank RGB24 VideoFrame with the given dimensions. */
export function createVideoFrame(width: number, height: number): VideoFrame {
  const buf = newHandleBuf();
  P10Error.check(ffiInt(p10_video_frame_create(buf, width, height)));
  return new VideoFrameImpl(buf);
}

// ------------------------------------------------------------------ //
// AudioFrame
// ------------------------------------------------------------------ //

export interface AudioFrame {
  /** Total number of samples per channel. */
  getSamplesCount(): bigint;
  /** Number of audio channels. */
  getChannelsCount(): bigint;
  /** Sample rate in Hz. */
  getSampleRate(): number;
  /** Duration in seconds. */
  getDurationSeconds(): number;
  /** Presentation timestamp. */
  getTime(): MediaTime;
  /** Set the presentation timestamp. */
  setTime(t: MediaTime): void;
  /**
   * Returns a non-owning Tensor view of the samples data (channels × samples).
   * The view is valid only while this AudioFrame is alive.
   * Do NOT call delete() on the returned Tensor.
   */
  getSamples(): Tensor;
  /** Releases the native frame handle. */
  delete(): void;
}

class AudioFrameImpl implements AudioFrame {
  readonly _buf: BigUint64Array;

  constructor(buf: BigUint64Array) {
    this._buf = buf;
  }

  getSamplesCount(): bigint {
    return ffiU64(p10_audio_frame_samples_count(readHandle(this._buf)));
  }
  getChannelsCount(): bigint {
    return ffiU64(p10_audio_frame_channels_count(readHandle(this._buf)));
  }
  getSampleRate(): number {
    return Number(ffiU64(p10_audio_frame_sample_rate(readHandle(this._buf))));
  }
  getDurationSeconds(): number {
    return p10_audio_frame_duration_seconds(readHandle(this._buf)) as number;
  }

  getTime(): MediaTime {
    const h = readHandle(this._buf);
    return {
      base: {
        num: ffiU64(p10_audio_frame_time_base_num(h)),
        den: ffiU64(p10_audio_frame_time_base_den(h)),
      },
      stamp: ffiU64(p10_audio_frame_time_stamp(h)),
    };
  }

  setTime(t: MediaTime): void {
    p10_audio_frame_set_time_parts(readHandle(this._buf), t.base.num, t.base.den, t.stamp);
  }

  getSamples(): Tensor {
    const sBuf = newHandleBuf();
    P10Error.check(ffiInt(p10_audio_frame_samples(readHandle(this._buf), sBuf)));
    return _wrapHandle(sBuf);
  }

  delete(): void {
    if (this._buf[0] !== 0n) {
      P10Error.check(ffiInt(p10_audio_frame_destroy(this._buf)));
    }
  }
}

/**
 * Creates an AudioFrame by copying data from a samples Tensor.
 * The tensor must have shape [channels, numSamples].
 */
export function createAudioFrame(samples: Tensor, sampleRate: number): AudioFrame {
  const buf = newHandleBuf();
  P10Error.check(ffiInt(p10_audio_frame_create(buf, Number(_getRawHandle(samples)), sampleRate)));
  return new AudioFrameImpl(buf);
}

// ------------------------------------------------------------------ //
// MediaCapture
// ------------------------------------------------------------------ //

export interface MediaCapture {
  /** Advances to the next frame. Returns false at end-of-stream. */
  nextFrame(): boolean;
  /** Decodes the current video frame. Caller must call delete(). */
  getVideo(): VideoFrame;
  /** Decodes the current audio frame. Caller must call delete(). */
  getAudio(): AudioFrame;
  /** Video width in pixels. */
  getVideoWidth(): number;
  /** Video height in pixels. */
  getVideoHeight(): number;
  /** Video frame rate as a rational {num, den}. */
  getVideoFrameRate(): Rational;
  /** Audio sample rate in Hz (0 if no audio stream). */
  getAudioSampleRate(): number;
  /** Number of audio channels (0 if no audio stream). */
  getAudioChannels(): number;
  /** Total video frame count, or -1n if unknown. */
  getVideoFrameCount(): bigint;
  /** Total duration in seconds, or -1 if unknown. */
  getDuration(): number;
  /** Closes the capture session. */
  close(): void;
}

class MediaCaptureImpl implements MediaCapture {
  private _buf: BigUint64Array;

  constructor(buf: BigUint64Array) {
    this._buf = buf;
  }

  nextFrame(): boolean {
    const hasFrame = new Int32Array(1);
    P10Error.check(ffiInt(p10_media_capture_next_frame(readHandle(this._buf), hasFrame)));
    return hasFrame[0] !== 0;
  }

  getVideo(): VideoFrame {
    const frameBuf = newHandleBuf();
    P10Error.check(ffiInt(p10_media_capture_get_video(readHandle(this._buf), frameBuf)));
    return new VideoFrameImpl(frameBuf);
  }

  getAudio(): AudioFrame {
    const frameBuf = newHandleBuf();
    P10Error.check(ffiInt(p10_media_capture_get_audio(readHandle(this._buf), frameBuf)));
    return new AudioFrameImpl(frameBuf);
  }

  getVideoWidth(): number {
    return ffiInt(p10_media_capture_video_width(readHandle(this._buf)));
  }
  getVideoHeight(): number {
    return ffiInt(p10_media_capture_video_height(readHandle(this._buf)));
  }

  getVideoFrameRate(): Rational {
    const h = readHandle(this._buf);
    return {
      num: ffiU64(p10_media_capture_video_frame_rate_num(h)),
      den: ffiU64(p10_media_capture_video_frame_rate_den(h)),
    };
  }

  getAudioSampleRate(): number {
    return p10_media_capture_audio_sample_rate(readHandle(this._buf)) as number;
  }
  getAudioChannels(): number {
    return Number(ffiU64(p10_media_capture_audio_channels(readHandle(this._buf))));
  }
  getVideoFrameCount(): bigint {
    return ffiU64(p10_media_capture_video_frame_count(readHandle(this._buf)));
  }
  getDuration(): number {
    return p10_media_capture_duration(readHandle(this._buf)) as number;
  }

  close(): void {
    if (this._buf[0] !== 0n) {
      P10Error.check(ffiInt(p10_media_capture_close(this._buf)));
    }
  }
}

/**
 * Opens a media file for reading (any container/codec FFmpeg can decode).
 * Call close() when done.
 */
export function openCapture(path: string): MediaCapture {
  const buf = newHandleBuf();
  const pathBuf = Buffer.from(`${path}\0`);
  P10Error.check(ffiInt(p10_media_capture_open(buf, pathBuf)));
  return new MediaCaptureImpl(buf);
}

// ------------------------------------------------------------------ //
// MediaWriter
// ------------------------------------------------------------------ //

export interface MediaWriter {
  /** Writes one video frame. */
  writeVideo(frame: VideoFrame): void;
  /** Writes one audio frame. */
  writeAudio(frame: AudioFrame): void;
  /** Closes the writer and flushes all pending data. */
  close(): void;
}

class MediaWriterImpl implements MediaWriter {
  private _buf: BigUint64Array;

  constructor(buf: BigUint64Array) {
    this._buf = buf;
  }

  writeVideo(frame: VideoFrame): void {
    P10Error.check(
      ffiInt(
        p10_media_writer_write_video(
          readHandle(this._buf),
          readHandle((frame as VideoFrameImpl)._buf),
        ),
      ),
    );
  }

  writeAudio(frame: AudioFrame): void {
    P10Error.check(
      ffiInt(
        p10_media_writer_write_audio(
          readHandle(this._buf),
          readHandle((frame as AudioFrameImpl)._buf),
        ),
      ),
    );
  }

  close(): void {
    if (this._buf[0] !== 0n) {
      P10Error.check(ffiInt(p10_media_writer_close(this._buf)));
    }
  }
}

/**
 * Opens a media file for writing.
 * Pass audioSampleRate=0 and audioChannels=0 to disable audio.
 * Call close() when done.
 */
export function openWriter(
  path: string,
  width: number,
  height: number,
  frameRate: Rational,
  audioSampleRate: number = 0,
  audioChannels: number = 0,
): MediaWriter {
  const buf = newHandleBuf();
  const pathBuf = Buffer.from(`${path}\0`);
  P10Error.check(
    ffiInt(
      p10_media_writer_open_ffi(
        buf,
        pathBuf,
        width,
        height,
        frameRate.num,
        frameRate.den,
        audioSampleRate,
        audioChannels,
      ),
    ),
  );
  return new MediaWriterImpl(buf);
}
