// Times the tlog receive path stage by stage, on the payloads a session sends:
// framing -> JSON.parse -> validate -> base64 -> zstd -> typed array -> render.
// Run with: node bindings/typescript/ptensor-view/bench-tlog.ts

import { zstdCompressSync } from 'node:zlib';
import { lineSplitter } from './packages/tlog/src/lineSplitter.ts';
import { parseIncoming } from './packages/tlog/src/protocol.ts';
import { base64ToBytes, bytesToBase64 } from '../ptensor-ts/src/base64.ts';
import { decodeTensorBlob, tensorFromJson } from '../ptensor-ts/src/tensorJson.ts';
import type { TensorJson } from '../ptensor-ts/src/tensorJson.ts';
import { computeStats } from './packages/view/src/stats.ts';
import { planeToRgba, imageMapping } from './packages/view/src/imageData.ts';

type Kind = 'gradient' | 'photo' | 'noise';

// Mirrors the three payload kinds of the C++ bench: a flat ramp that zstd
// crushes, a photo-like ramp+noise that it barely touches, and pure noise.
function makeImage(h: number, w: number, kind: Kind): Uint8Array {
  const out = new Uint8Array(h * w * 3);
  if (kind === 'gradient') {
    for (let i = 0; i < out.length; i++) {
      out[i] = (i >> 4) & 0xff;
    }
    return out;
  }
  const noise = new Uint8Array(out.length);
  for (let off = 0; off < noise.length; off += 65536) {
    crypto.getRandomValues(noise.subarray(off, Math.min(off + 65536, noise.length)));
  }
  if (kind === 'noise') {
    return noise;
  }
  for (let i = 0; i < out.length; i++) {
    out[i] = (((i >> 6) & 0xff) + (noise[i] % 41)) & 0xff;
  }
  return out;
}

function makeCase(h: number, w: number, kind: Kind) {
  const raw = makeImage(h, w, kind);
  const compressed = new Uint8Array(zstdCompressSync(raw));
  const json: TensorJson = {
    dtype: 'uint8',
    shape: [h, w, 3],
    stride: [w * 3, 3, 1],
    size_bytes: raw.length,
    encoding: 'base64+zstd',
    blob: bytesToBase64(compressed),
  };
  return {
    label: `${h}x${w} ${kind}`,
    line: JSON.stringify({ name: 'bench', tensor: json }),
    json,
    rawBytes: raw.length,
  };
}

function bench(name: string, bytes: number, fn: () => unknown, minMs = 400): void {
  for (let i = 0; i < 3; i++) {
    fn();
  }
  let iterations = 0;
  const start = performance.now();
  let elapsed = 0;
  while (elapsed < minMs) {
    fn();
    iterations++;
    elapsed = performance.now() - start;
  }
  const perIter = elapsed / iterations;
  const mbs = bytes / 1e6 / (perIter / 1000);
  console.log(
    `${name.padEnd(40)} ${perIter.toFixed(3).padStart(9)} ms ${mbs.toFixed(0).padStart(6)} MB/s (${iterations} it)`
  );
}

// Baselines for the two per-pixel loops: same result, but reading the typed
// array directly instead of through `elementAt` (an instanceof test per
// element) and skipping the identity map that uint8 data does not need.
function statsDirect(data: ArrayLike<number>): { min: number; max: number } {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (v < min) {
      min = v;
    }
    if (v > max) {
      max = v;
    }
  }
  return { min, max };
}

function rgbaDirect(data: Uint8Array, width: number, height: number): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let i = 0, j = 0; i < width * height; i++, j += 3) {
    const idx = i * 4;
    rgba[idx] = data[j];
    rgba[idx + 1] = data[j + 1];
    rgba[idx + 2] = data[j + 2];
    rgba[idx + 3] = 255;
  }
  return rgba;
}

const CHUNK = 64 * 1024; // what a loopback socket hands node per 'data' event

// Baseline for the splitter: scan the raw chunk for the newline and decode a
// line once, instead of decoding every chunk into an ever growing string that
// is then re-scanned and re-sliced from index 0 on each 'data' event.
function byteLineSplitter(): (chunk: Uint8Array) => string[] {
  const decoder = new TextDecoder('utf-8');
  let pending: Uint8Array[] = [];
  let pendingBytes = 0;
  return (chunk: Uint8Array): string[] => {
    const lines: string[] = [];
    let start = 0;
    for (;;) {
      const nl = chunk.indexOf(10, start);
      if (nl === -1) {
        break;
      }
      pending.push(chunk.subarray(start, nl));
      pendingBytes += nl - start;
      const joined = new Uint8Array(pendingBytes);
      let at = 0;
      for (const part of pending) {
        joined.set(part, at);
        at += part.length;
      }
      lines.push(decoder.decode(joined));
      pending = [];
      pendingBytes = 0;
      start = nl + 1;
    }
    if (start < chunk.length) {
      const tail = chunk.subarray(start);
      pending.push(tail);
      pendingBytes += tail.length;
    }
    return lines;
  };
}

const cases = [
  makeCase(480, 640, 'gradient'),
  makeCase(480, 640, 'photo'),
  makeCase(480, 640, 'noise'),
  makeCase(720, 1280, 'photo'),
];

for (const testCase of cases) {
  const { label, line, json, rawBytes } = testCase;
  const wire = new TextEncoder().encode(line + '\n');
  const tensor = tensorFromJson(json);

  console.log(
    `\n== ${label}: ${(rawBytes / 1e6).toFixed(2)} MB raw -> ${(wire.length / 1e6).toFixed(2)} MB on the wire`
  );

  bench('lineSplitter (64 KiB chunks)', rawBytes, () => {
    const split = lineSplitter(256 * 1024 * 1024);
    for (let off = 0; off < wire.length; off += CHUNK) {
      split(wire.subarray(off, Math.min(off + CHUNK, wire.length)));
    }
  });
  bench('byteLineSplitter (64 KiB chunks)', rawBytes, () => {
    const split = byteLineSplitter();
    for (let off = 0; off < wire.length; off += CHUNK) {
      split(wire.subarray(off, Math.min(off + CHUNK, wire.length)));
    }
  });
  bench('JSON.parse(line)', rawBytes, () => JSON.parse(line));
  const parsed = JSON.parse(line);
  bench('parseIncoming (validate)', rawBytes, () => parseIncoming(parsed));
  bench('base64ToBytes (Buffer)', rawBytes, () => base64ToBytes(json.blob));
  bench('base64ToBytes (atob fallback)', rawBytes, () => {
    const bin = atob(json.blob);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) {
      out[i] = bin.charCodeAt(i);
    }
    return out;
  });
  bench('decodeTensorBlob (base64 + fzstd)', rawBytes, () => decodeTensorBlob(json));
  bench('tensorFromJson (+ copy)', rawBytes, () => tensorFromJson(json));
  bench('computeStats', rawBytes, () => computeStats(tensor.data));
  bench('computeStats (direct typed-array)', rawBytes, () => statsDirect(tensor.data));
  const stats = computeStats(tensor.data);
  const plane = {
    width: json.shape[1],
    height: json.shape[0],
    channels: 3,
    layout: 'interleaved' as const,
  };
  bench('planeToRgba', rawBytes, () =>
    planeToRgba(tensor.data, 0, plane, imageMapping('uint8', stats))
  );
  bench('planeToRgba (direct uint8 copy)', rawBytes, () =>
    rgbaDirect(tensor.data as Uint8Array, plane.width, plane.height)
  );

  // The host -> webview hop: electrobun's RPC and VS Code's postMessage both
  // serialize the payload, so the base64 blob crosses as JSON a second time.
  const payload = { sessionId: 's', name: 'bench', tensor: json, receivedAt: 0 };
  bench('IPC hop (stringify + parse payload)', rawBytes, () =>
    JSON.parse(JSON.stringify(payload))
  );
  bench('FRAME TOTAL (split+parse+decode+draw)', rawBytes, () => {
    const split = lineSplitter(256 * 1024 * 1024);
    let lines: string[] = [];
    for (let off = 0; off < wire.length; off += CHUNK) {
      const out = split(wire.subarray(off, Math.min(off + CHUNK, wire.length)));
      if (out !== 'overflow') {
        lines = lines.concat(out);
      }
    }
    const message = parseIncoming(JSON.parse(lines[0]));
    if (message.kind !== 'tensor-message') {
      throw new Error('unexpected message');
    }
    const view = tensorFromJson(message.tensor);
    return planeToRgba(view.data, 0, plane, imageMapping('uint8', computeStats(view.data)));
  });
}
