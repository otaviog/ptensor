// What crosses between the decode worker and the window, and the two pure
// functions that put a tensor into that form and take it back out.
//
// Kept out of the worker shell so it can be tested directly: the worker itself
// is bundled into a string at build time, which nothing can import.

import {
    asDType,
    tensorFromJson,
    viewNumericArray,
    type Tensor,
    type TensorJson,
} from 'ptensor-ts';

export interface DecodeRequest {
    /** Matches the response to its caller. */
    id: number;
    /** Where the worker fetches the `TensorJson`, token and all. */
    url: string;
}

/**
 * A decoded tensor in a form the worker can hand over rather than copy: the
 * buffer is transferred, so a 222 MiB tensor crosses for free.
 */
export interface DecodedTransfer {
    dtype: string;
    /**
     * The dtype whose typed array matches `buffer`. Only float16 differs: it
     * has no native typed array, so it arrives widened to float32 while
     * `dtype` keeps saying what the tensor is.
     */
    viewDtype: string;
    shape: number[];
    stride: number[];
    buffer: ArrayBuffer;
}

export type DecodeResponse =
    | ({ id: number; ok: true } & DecodedTransfer)
    | { id: number; ok: false; error: string };

/** Decodes a `TensorJson` into a transferable form. Runs in the worker. */
export function toTransfer(json: TensorJson): DecodedTransfer {
    const tensor = tensorFromJson(json);
    const data = tensor.data;
    // The buffer is handed over whole, so it has to be the tensor's alone.
    const exact = data.byteOffset === 0 && data.byteLength === data.buffer.byteLength;
    const buffer = exact
        ? (data.buffer as ArrayBuffer)
        : data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
    return {
        dtype: tensor.dtype,
        viewDtype: tensor.dtype === 'float16' ? 'float32' : tensor.dtype,
        shape: tensor.shape,
        stride: tensor.stride,
        buffer: buffer as ArrayBuffer,
    };
}

/** Rebuilds the tensor around a transferred buffer. Runs in the window. */
export function fromTransfer(transfer: DecodedTransfer): Tensor {
    const dtype = asDType(transfer.dtype);
    const viewDtype = asDType(transfer.viewDtype);
    if (!dtype || !viewDtype) {
        throw new Error(
            `the decoder reported an unknown dtype '${transfer.dtype}'/'${transfer.viewDtype}'`
        );
    }
    return {
        dtype,
        shape: transfer.shape,
        stride: transfer.stride,
        data: viewNumericArray(viewDtype, transfer.buffer),
    };
}

/** Fetches one tensor's JSON. Used by the worker and the fallback alike. */
export async function fetchTensorJson(url: string): Promise<TensorJson> {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`the tensor request answered ${response.status}`);
    }
    return (await response.json()) as TensorJson;
}
