import type { ImagePlane } from './resolveView';
import { elementAt, type NumericArray } from 'ptensor-ts';
import type { TensorStats } from './stats';
import { isFloatDtype, type DTypeString } from 'ptensor-ts';

export interface ImageMapping {
    lo: number;
    scale: number;
}

/**
 * Display mapping for a plane: uint8-style dtypes map 0..255 directly, floats
 * stretch by the tensor min/max so the full range is visible.
 */
export function imageMapping(dtype: DTypeString, stats: TensorStats): ImageMapping {
    const stretch = isFloatDtype(dtype) || dtype === 'int8' || dtype === 'int16'
        || dtype === 'int32' || dtype === 'int64';
    const lo = stretch ? stats.min : 0;
    const hi = stretch ? stats.max : 255;
    const range = hi - lo;
    return { lo, scale: range > 0 ? 255 / range : 1 };
}

/**
 * Converts one image plane of a (possibly batched) tensor into an RGBA buffer
 * ready for `new ImageData(...)`. `offset` is the element index where this
 * plane starts; `layout` selects interleaved (HWC) vs planar (CHW) addressing.
 */
export function planeToRgba(
    array: NumericArray,
    offset: number,
    plane: ImagePlane,
    mapping: ImageMapping
): Uint8ClampedArray<ArrayBuffer> {
    const { width, height, channels, layout } = plane;
    const { lo, scale } = mapping;
    const pixels = width * height;
    const rgba = new Uint8ClampedArray(pixels * 4);

    // The common case -- an interleaved uint8 image, whose mapping is the
    // identity -- is a straight copy. Going through the generic path instead
    // costs an `instanceof` test and a `Math.round` per channel, six times the
    // time on a 720p frame.
    const bytes =
        array instanceof Uint8Array || array instanceof Uint8ClampedArray ? array : null;
    if (bytes !== null && layout === 'interleaved' && lo === 0 && scale === 1) {
        if (channels === 1) {
            for (let i = 0, src = offset; i < pixels; i++, src++) {
                const idx = i * 4;
                const v = bytes[src];
                rgba[idx] = v;
                rgba[idx + 1] = v;
                rgba[idx + 2] = v;
                rgba[idx + 3] = 255;
            }
            return rgba;
        }
        if (channels === 3 || channels === 4) {
            for (let i = 0, src = offset; i < pixels; i++, src += channels) {
                const idx = i * 4;
                rgba[idx] = bytes[src];
                rgba[idx + 1] = bytes[src + 1];
                rgba[idx + 2] = bytes[src + 2];
                rgba[idx + 3] = channels === 4 ? bytes[src + 3] : 255;
            }
            return rgba;
        }
    }

    // Generic path: any dtype, any layout, and a mapping that stretches the
    // values. The addressing is picked once, not per channel read.
    const interleaved = layout === 'interleaved';
    const chan = (pixel: number, c: number): number =>
        elementAt(array, interleaved ? offset + pixel * channels + c : offset + c * pixels + pixel);
    const map = (v: number) => Math.round((v - lo) * scale);

    for (let i = 0; i < pixels; i++) {
        const idx = i * 4;
        if (channels === 1) {
            const v = map(chan(i, 0));
            rgba[idx] = v;
            rgba[idx + 1] = v;
            rgba[idx + 2] = v;
            rgba[idx + 3] = 255;
        } else {
            rgba[idx] = map(chan(i, 0));
            rgba[idx + 1] = map(chan(i, 1));
            rgba[idx + 2] = map(chan(i, 2));
            rgba[idx + 3] = channels === 4 ? map(chan(i, 3)) : 255;
        }
    }
    return rgba;
}
