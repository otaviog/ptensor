/**
 * Portable base64 <-> bytes, dependency-free.
 *
 * Three paths, fastest first: the engine's native `Uint8Array.fromBase64` /
 * `toBase64` (one allocation, no intermediate string), Node/Bun's `Buffer`,
 * then a plain `atob`/`btoa` loop for engines that have neither.
 *
 * The loop is the floor on purpose. The idiomatic-looking
 * `atob(b64).split('').map((c) => c.charCodeAt(0))` allocates an array of one
 * character string per byte -- measured at ~17 bytes of heap per decoded byte,
 * and ten times slower than the loop -- which is what turns a 200MB tensor
 * into gigabytes of webview memory.
 */

declare function atob(data: string): string;
declare function btoa(data: string): string;

declare const Buffer:
  | { from(input: string, enc: string): { buffer: ArrayBuffer; byteOffset: number; byteLength: number };
      from(input: Uint8Array): { toString(enc: string): string }; }
  | undefined;

/**
 * The TC39 base64 accessors (Safari 18.2+, Node 22+). Typed here rather than
 * relied on from lib.d.ts, which does not carry them in every TS release we
 * build against.
 */
type Base64Statics = { fromBase64?(input: string): Uint8Array };
type Base64Methods = { toBase64?(): string };

const fromBase64 = (Uint8Array as unknown as Base64Statics).fromBase64;
const toBase64 = (Uint8Array.prototype as unknown as Base64Methods).toBase64;

export function base64ToBytes(b64: string): Uint8Array {
  if (typeof fromBase64 === 'function') {
    return fromBase64.call(Uint8Array, b64);
  }
  if (typeof Buffer !== 'undefined') {
    const buf = Buffer.from(b64, 'base64') as { buffer: ArrayBuffer; byteOffset: number; byteLength: number };
    return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  }
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) {
    out[i] = bin.charCodeAt(i);
  }
  return out;
}

export function bytesToBase64(bytes: Uint8Array): string {
  if (typeof toBase64 === 'function') {
    return toBase64.call(bytes);
  }
  if (typeof Buffer !== 'undefined') {
    return (Buffer.from(bytes) as { toString(enc: string): string }).toString('base64');
  }
  // Chunked: `String.fromCharCode(...bytes)` would spread the whole array onto
  // the call stack and overflow on anything large.
  const CHUNK = 8192;
  let bin = '';
  for (let at = 0; at < bytes.length; at += CHUNK) {
    bin += String.fromCharCode(...bytes.subarray(at, at + CHUNK));
  }
  return btoa(bin);
}
