import { base64ToBytes, bytesToBase64 } from './base64';
import { asDType, type DTypeString, dtypeSizeBytes } from './dtype';
import { type NumericArray, viewNumericArray } from './numericArray';
import { parseTensorJson, type TensorJson } from './tensorJson';

/**
 * A materialized, JS-owned tensor: plain data with no native handle. This is
 * what `parse`/`tensorFromJson` produce and what the viewer consumes. The FFI
 * binding's `PTensor.toTensor()` also yields this (an explicit copy out of
 * native memory). Strides are in element counts.
 */
export interface Tensor {
  dtype: DTypeString;
  shape: number[];
  stride: number[];
  data: NumericArray;
}

/** Row-major (C-contiguous) strides for a shape, in element counts. */
export function contiguousStride(shape: number[]): number[] {
  const stride = new Array<number>(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    stride[i] = acc;
    acc *= shape[i];
  }
  return stride;
}

/** Element count for a shape. */
export function numElements(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

/** Decodes a `TensorJson` (base64 blob) into a materialized `Tensor`. */
export function tensorFromJson(json: TensorJson): Tensor {
  const dtype = asDType(json.dtype);
  if (!dtype) {
    throw new Error(`Unknown dtype '${json.dtype}' in tensor JSON.`);
  }
  const bytes = base64ToBytes(json.blob);
  const elemSize = dtypeSizeBytes[dtype];
  if (bytes.byteLength % elemSize !== 0) {
    throw new Error(
      `Blob length ${bytes.byteLength} is not a multiple of ${elemSize} for dtype '${dtype}'.`,
    );
  }
  // Copy into a fresh, element-aligned buffer: the base64 bytes may be offset
  // inside a larger Buffer, which a typed-array view can't straddle safely.
  const aligned = bytes.slice();
  const data = viewNumericArray(dtype, aligned.buffer, aligned.byteOffset, bytes.byteLength / elemSize);
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
    blob: bytesToBase64(bytes),
  };
}
