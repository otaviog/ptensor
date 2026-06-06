import { asDType, base64ToBytes, type TensorJson, viewNumericArray } from 'ptensor-ts';
import { type DTypeString, type NumericArray, type TensorView } from './types';

// The transport form (`TensorJson` = `{dtype, shape, stride, blob}`, identical
// to what `p10::to_json_debug` emits) and the raw byte decode are owned by
// ptensor-ts. This module owns only the view-specific concerns: assembling a
// renderable `TensorView` (bigint shapes) and decoding float16 -> Float32 for
// display (there is no native float16 typed array).

/** Decodes a `TensorJson` into a renderable `TensorView` (float16 -> Float32Array). */
export function fromTensorJson(json: TensorJson, name?: string): TensorView {
    // ptensor-ts keeps `dtype` loose (string) on the wire; validate/narrow here.
    const dtype = asDType(json.dtype);
    if (!dtype) {
        throw new Error(`Unknown dtype '${json.dtype}' in tensor JSON.`);
    }
    return {
        name,
        dtype,
        shape: json.shape.map(BigInt),
        stride: json.stride.map(BigInt),
        array: bytesToTyped(base64ToArrayBuffer(json.blob), dtype),
    };
}

/**
 * Decodes raw bytes into the NumericArray for a dtype. float16 is widened to
 * Float32Array for rendering; every other dtype is a plain typed-array view
 * (delegated to ptensor-ts).
 */
export function bytesToTyped(buffer: ArrayBuffer, dtype: DTypeString): NumericArray {
    if (dtype === 'float16') {
        return float16ToFloat32(new Uint16Array(buffer));
    }
    return viewNumericArray(dtype, buffer);
}

/** Base64 -> a fresh, element-aligned ArrayBuffer (delegates decode to ptensor-ts). */
export function base64ToArrayBuffer(base64: string): ArrayBuffer {
    // `.slice()` copies into a zero-offset buffer: Buffer-backed bytes from
    // ptensor-ts may sit at a non-zero offset that typed-array views can't span.
    return base64ToBytes(base64).slice().buffer;
}

function float16ToFloat32(input: Uint16Array): Float32Array {
    const out = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) {
        const h = input[i];
        const sign = (h & 0x8000) >> 15;
        const exp = (h & 0x7c00) >> 10;
        const frac = h & 0x03ff;
        if (exp === 0) {
            out[i] = (sign ? -1 : 1) * 2 ** -14 * (frac / 1024);
        } else if (exp === 31) {
            out[i] = frac ? NaN : sign ? -Infinity : Infinity;
        } else {
            out[i] = (sign ? -1 : 1) * 2 ** (exp - 15) * (1 + frac / 1024);
        }
    }
    return out;
}
