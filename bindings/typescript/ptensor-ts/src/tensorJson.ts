import { decompress } from 'fzstd';

import { P10Error } from './p10error';
import { bytesToTyped, Tensor } from './tensor';

import { base64ToBytes, bytesToBase64 } from './base64';
import { asDType, dtypeSizeBytes } from './dtype';

/**
 * How `blob` is encoded. `base64` is the raw little-endian element bytes;
 * `base64+zstd` is those bytes zstd-compressed, then base64. Mirrors the
 * strings emitted by `p10::Json` in `src/core/json.cpp`.
 */
export type TensorEncoding = 'base64' | 'base64+zstd';

const ENCODINGS: readonly string[] = ['base64', 'base64+zstd'];

/**
 * The wire format emitted by `p10::to_json_debug` and by binaries that print a
 * tensor to stdout: `{dtype, shape, stride, size_bytes, encoding, blob}`.
 * `size_bytes` is the *uncompressed* element-byte count, so a reader can size
 * its buffer before touching `blob`.
 */
export type TensorJson = {
  dtype: string;
  shape: number[];
  stride: number[];
  size_bytes: number;
  encoding: TensorEncoding;
  blob: string;
};

/**
 * Extracts and parses a `TensorJson` from a raw debugger / stdout string. The
 * input may be wrapped (e.g. an LLDB summary `(const char *) $0 = 0x.. "{...}"`),
 * so we slice between the first `{` and last `}` and undo C-string escaping.
 */
export function parseTensorJson(rawResult: string): TensorJson {
  const start = rawResult.indexOf('{');
  const end = rawResult.lastIndexOf('}');
  if (start < 0 || end <= start) {
    throw new P10Error(
      `Could not find a JSON object in evaluate result: ${rawResult.slice(0, 200)}`,
    );
  }
  const jsonText = rawResult
    .slice(start, end + 1)
    .replace(/\\"/g, '"')
    .replace(/\\\\/g, '\\');

  let obj: unknown;
  try {
    obj = JSON.parse(jsonText);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new P10Error(`Could not parse tensor JSON: ${msg}`);
  }
  const json = validateTensorJson(obj);
  return {
    ...json,
    shape: json.shape.map((n) => Number(n)),
    stride: json.stride.map((n) => Number(n)),
  };
}

export function validateTensorJson(value: unknown): TensorJson {
  if (typeof value !== 'object' || value === null) {
    throw new P10Error(`Tensor JSON is not an object: ${String(value)}`);
  }

  const valueDict = value as Record<string, unknown>;
  if (typeof valueDict.dtype !== 'string') {
    throw new P10Error(
      `Tensor JSON is missing 'dtype' field or it is not a string: ${String(value)}`,
    );
  }
  if (!Array.isArray(valueDict.shape)) {
    throw new P10Error(
      `Tensor JSON is missing 'shape' field or it is not an array: ${String(value)}`,
    );
  }

  if (!Array.isArray(valueDict.stride)) {
    throw new P10Error(
      `Tensor JSON is missing 'stride' field or it is not an array: ${String(value)}`,
    );
  }

  if (typeof valueDict.size_bytes !== 'number' || !Number.isInteger(valueDict.size_bytes)
    || valueDict.size_bytes < 0) {
    throw new P10Error(
      `Tensor JSON is missing 'size_bytes' field or it is not a non-negative integer: ${String(value)}`,
    );
  }

  if (typeof valueDict.encoding !== 'string' || !ENCODINGS.includes(valueDict.encoding)) {
    throw new P10Error(
      `Tensor JSON has an unknown 'encoding' field, expected one of ${ENCODINGS.join(', ')}: ${String(valueDict.encoding)}`,
    );
  }

  if (typeof valueDict.blob !== 'string') {
    throw new P10Error(`Tensor JSON is missing 'blob' field or it is not a string: ${String(value)}`);
  }

  return value as TensorJson;
}

/**
 * Undoes `encoding`, returning the raw little-endian element bytes.
 *
 * `decompress` is deliberately called without an output buffer. Passing one
 * would have it return that whole buffer rather than the written slice, so a
 * frame that decompresses short would come back zero-padded and slip past the
 * `size_bytes` check in `tensorFromJson`. It buys nothing either: a single
 * frame that declares its content size -- what zstd writes by default -- is
 * already decompressed straight into one exactly-sized buffer.
 */
export function decodeTensorBlob(json: TensorJson): Uint8Array {
  const bytes = base64ToBytes(json.blob);
  if (json.encoding !== 'base64+zstd') {
    return bytes;
  }
  try {
    return decompress(bytes);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new P10Error(`Could not zstd-decompress the tensor blob: ${msg}`);
  }
}

/**
 * Decodes a `TensorJson` into a materialized `Tensor`, undoing both the base64
 * and, for `base64+zstd`, the compression. A float16 payload is widened to a
 * `Float32Array` (see `bytesToTyped`) while `dtype` keeps saying `float16`.
 */
export function tensorFromJson(json: TensorJson): Tensor {
  const dtype = asDType(json.dtype);
  if (!dtype) {
    throw new P10Error(`Unknown dtype '${json.dtype}' in tensor JSON.`);
  }
  const bytes = decodeTensorBlob(json);
  if (bytes.byteLength !== json.size_bytes) {
    throw new P10Error(
      `Blob decodes to ${bytes.byteLength} bytes but 'size_bytes' says ${json.size_bytes}.`,
    );
  }
  const elemSize = dtypeSizeBytes[dtype];
  if (bytes.byteLength % elemSize !== 0) {
    throw new P10Error(
      `Blob length ${bytes.byteLength} is not a multiple of ${elemSize} for dtype '${dtype}'.`,
    );
  }
  // A typed-array view can only be taken over a buffer the bytes cover exactly
  // and from the start -- `Buffer.from` hands back a view into a pooled, larger
  // one. Copy only then: at tensor sizes this is hundreds of megabytes.
  const exact = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength;
  const data = bytesToTyped(exact ? (bytes.buffer as ArrayBuffer) : bytes.slice().buffer, dtype);
  return { dtype, shape: json.shape, stride: json.stride, data };
}

/** Parses raw debugger/stdout text straight into a materialized `Tensor`. */
export function parse(rawResult: string): Tensor {
  return tensorFromJson(parseTensorJson(rawResult));
}

/** Encodes a materialized `Tensor` back to the `TensorJson` wire format. */
export function tensorToJson(tensor: Tensor): TensorJson {
  const bytes = new Uint8Array(
    tensor.data.buffer,
    tensor.data.byteOffset,
    tensor.data.byteLength,
  );
  return {
    dtype: tensor.dtype,
    shape: tensor.shape,
    stride: tensor.stride,
    size_bytes: bytes.byteLength,
    encoding: 'base64',
    blob: bytesToBase64(bytes),
  };
}
