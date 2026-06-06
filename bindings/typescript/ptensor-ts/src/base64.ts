/**
 * Portable base64 <-> bytes, dependency-free. Prefers Node/Bun `Buffer` when
 * present (fastest), falls back to `atob`/`btoa` for browsers (the webview).
 */

declare function atob(data: string): string;
declare function btoa(data: string): string;

declare const Buffer:
  | { from(input: string, enc: string): { buffer: ArrayBuffer; byteOffset: number; byteLength: number };
      from(input: Uint8Array): { toString(enc: string): string }; }
  | undefined;

export function base64ToBytes(b64: string): Uint8Array {
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
  if (typeof Buffer !== 'undefined') {
    return (Buffer.from(bytes) as { toString(enc: string): string }).toString('base64');
  }
  let bin = '';
  for (let i = 0; i < bytes.length; i++) {
    bin += String.fromCharCode(bytes[i]);
  }
  return btoa(bin);
}
