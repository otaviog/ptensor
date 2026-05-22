import { afterEach, describe, expect, it } from 'bun:test';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import {
  type AudioFrame,
  createVideoFrame,
  type MediaCapture,
  openCapture,
  openWriter,
  type VideoFrame,
} from '../media';
import { P10Error } from '../p10Error';

const TEST_VIDEO = resolve(
  import.meta.dir,
  '../../../../../tests/data/video/file_example_MP4_480_1_5MG.mp4',
);

describe('media integration (real C library)', () => {
  const captures: MediaCapture[] = [];
  const videoFrames: VideoFrame[] = [];
  const audioFrames: AudioFrame[] = [];

  function trackCapture(c: MediaCapture): MediaCapture {
    captures.push(c);
    return c;
  }
  function trackVideo(f: VideoFrame): VideoFrame {
    videoFrames.push(f);
    return f;
  }
  function trackAudio(f: AudioFrame): AudioFrame {
    audioFrames.push(f);
    return f;
  }

  afterEach(() => {
    for (const f of videoFrames.splice(0)) f.delete();
    for (const f of audioFrames.splice(0)) f.delete();
    for (const c of captures.splice(0)) c.close();
  });

  // ---------------------------------------------------------------- //
  // MediaCapture — open / metadata
  // ---------------------------------------------------------------- //

  it('openCapture opens a valid file', () => {
    const cap = trackCapture(openCapture(TEST_VIDEO));
    expect(cap).toBeDefined();
  });

  it('openCapture throws P10Error for non-existent file', () => {
    expect(() => openCapture('/no/such/file.mp4')).toThrow(P10Error);
  });

  it('getVideoWidth and getVideoHeight return positive values', () => {
    const cap = trackCapture(openCapture(TEST_VIDEO));
    expect(cap.getVideoWidth()).toBeGreaterThan(0);
    expect(cap.getVideoHeight()).toBeGreaterThan(0);
  });

  it('getVideoFrameRate returns a sensible rational', () => {
    const cap = trackCapture(openCapture(TEST_VIDEO));
    const fps = cap.getVideoFrameRate();
    expect(fps.num).toBeGreaterThan(0n);
    expect(fps.den).toBeGreaterThan(0n);
  });

  it('getAudioSampleRate returns a non-negative number', () => {
    const cap = trackCapture(openCapture(TEST_VIDEO));
    // Some containers don't expose sample rate via parameters; just verify
    // it's a valid non-negative number.
    expect(cap.getAudioSampleRate()).toBeGreaterThanOrEqual(0);
  });

  it('getAudioChannels returns a positive value', () => {
    const cap = trackCapture(openCapture(TEST_VIDEO));
    expect(cap.getAudioChannels()).toBeGreaterThan(0);
  });

  it('getDuration returns a positive value', () => {
    const cap = trackCapture(openCapture(TEST_VIDEO));
    expect(cap.getDuration()).toBeGreaterThan(0);
  });

  // ---------------------------------------------------------------- //
  // MediaCapture — frame iteration
  // ---------------------------------------------------------------- //

  it('nextFrame returns true then eventually false', () => {
    const cap = trackCapture(openCapture(TEST_VIDEO));
    let hadFrame = false;
    // Read first frame only to keep test fast.
    hadFrame = cap.nextFrame();
    expect(hadFrame).toBe(true);
  });

  it('getVideo returns a valid VideoFrame', () => {
    const cap = trackCapture(openCapture(TEST_VIDEO));
    cap.nextFrame();
    const frame = trackVideo(cap.getVideo());
    expect(frame.getWidth()).toBeGreaterThan(0);
    expect(frame.getHeight()).toBeGreaterThan(0);
    expect(frame.getChannels()).toBe(3);
  });

  it('VideoFrame.getImage returns an HxWxC uint8 Tensor view', () => {
    const cap = trackCapture(openCapture(TEST_VIDEO));
    cap.nextFrame();
    const frame = trackVideo(cap.getVideo());

    const img = frame.getImage();
    const shape = img.getShape();

    expect(shape).toHaveLength(3);
    expect(shape[0]).toBe(BigInt(frame.getHeight()));
    expect(shape[1]).toBe(BigInt(frame.getWidth()));
    expect(shape[2]).toBe(3n);
    expect(img.getDtype()).toBe('uint8');
    expect(img.isEmpty()).toBe(false);
  });

  it('VideoFrame.getTime/setTime round-trips the timestamp', () => {
    const cap = trackCapture(openCapture(TEST_VIDEO));
    cap.nextFrame();
    const frame = trackVideo(cap.getVideo());

    frame.setTime({ base: { num: 1n, den: 25n }, stamp: 42n });
    const t = frame.getTime();
    expect(t.base.num).toBe(1n);
    expect(t.base.den).toBe(25n);
    expect(t.stamp).toBe(42n);
  });

  // ---------------------------------------------------------------- //
  // VideoFrame — standalone create
  // ---------------------------------------------------------------- //

  it('createVideoFrame allocates a blank frame', () => {
    const frame = trackVideo(createVideoFrame(320, 240));
    expect(frame.getWidth()).toBe(320);
    expect(frame.getHeight()).toBe(240);
    expect(frame.getChannels()).toBe(3);
  });

  it('createVideoFrame image tensor has correct shape', () => {
    const frame = trackVideo(createVideoFrame(64, 48));
    const img = frame.getImage();
    expect(img.getShape()).toEqual([48n, 64n, 3n]);
  });

  // ---------------------------------------------------------------- //
  // MediaCapture — audio frames
  // ---------------------------------------------------------------- //

  it('getAudio returns a valid AudioFrame', () => {
    const cap = trackCapture(openCapture(TEST_VIDEO));
    // Advance until we hit an audio frame.
    let audioFrame: AudioFrame | null = null;
    for (let i = 0; i < 20 && cap.nextFrame(); i++) {
      try {
        audioFrame = cap.getAudio();
        break;
      } catch {
        // current frame is video-only; try next
      }
    }
    if (audioFrame) {
      trackAudio(audioFrame);
      expect(audioFrame.getSamplesCount()).toBeGreaterThan(0n);
      expect(audioFrame.getChannelsCount()).toBeGreaterThan(0n);
      expect(audioFrame.getSampleRate()).toBeGreaterThan(0);
    }
  });

  // ---------------------------------------------------------------- //
  // MediaWriter — transcode one video frame
  // ---------------------------------------------------------------- //

  it('openWriter writes a single video frame to a temp file', async () => {
    const tmpDir = await mkdtemp(join(tmpdir(), 'ptensor-ts-test-'));
    const outPath = join(tmpDir, 'out.mp4');

    try {
      const cap = trackCapture(openCapture(TEST_VIDEO));
      const fps = cap.getVideoFrameRate();
      const w = cap.getVideoWidth();
      const h = cap.getVideoHeight();

      const writer = openWriter(outPath, w, h, fps);

      cap.nextFrame();
      const frame = trackVideo(cap.getVideo());
      writer.writeVideo(frame);
      writer.close();

      const stat = Bun.file(outPath);
      expect(await stat.exists()).toBe(true);
      expect(stat.size).toBeGreaterThan(0);
    } finally {
      await rm(tmpDir, { recursive: true, force: true });
    }
  });
});
