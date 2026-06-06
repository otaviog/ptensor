import type { ImagePlane } from './resolveView';
import { elementAt, type NumericArray } from './types';
import type { TensorStats } from './stats';
import { isFloatDtype, type DTypeString } from './types';

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
): Uint8ClampedArray {
    const { width, height, channels, layout } = plane;
    const { lo, scale } = mapping;
    const rgba = new Uint8ClampedArray(width * height * 4);
    const planeSize = width * height;

    const chan = (x: number, y: number, c: number): number => {
        const i = layout === 'interleaved'
            ? offset + (y * width + x) * channels + c
            : offset + c * planeSize + y * width + x;
        return elementAt(array, i);
    };
    const map = (v: number) => Math.round((v - lo) * scale);

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            if (channels === 1) {
                const v = map(chan(x, y, 0));
                rgba[idx] = v;
                rgba[idx + 1] = v;
                rgba[idx + 2] = v;
                rgba[idx + 3] = 255;
            } else {
                rgba[idx] = map(chan(x, y, 0));
                rgba[idx + 1] = map(chan(x, y, 1));
                rgba[idx + 2] = map(chan(x, y, 2));
                rgba[idx + 3] = channels === 4 ? map(chan(x, y, 3)) : 255;
            }
        }
    }
    return rgba;
}
