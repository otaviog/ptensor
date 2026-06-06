import type { TensorView } from './types';

/**
 * Synthetic tensors used by the dev playground and the in-editor demo command.
 * One per branch of `resolveView`:
 * tables (incl. int64), grayscale/RGB images (planar + interleaved), batched
 * NCHW/NHWC, and a large non-image buffer. Mirrors the C++ `vscode_viewer_demo`
 * driver so the offline and live paths exercise the same panel code.
 */

function view(
    name: string,
    shape: number[],
    dtype: TensorView['dtype'],
    array: TensorView['array']
): TensorView {
    return {
        name,
        shape: shape.map(BigInt),
        stride: contiguousStride(shape).map(BigInt),
        dtype,
        array,
    };
}

function contiguousStride(shape: number[]): number[] {
    const stride = new Array<number>(shape.length).fill(1);
    for (let i = shape.length - 2; i >= 0; i--) {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
    return stride;
}

function grayGradient(h: number, w: number): Float32Array {
    const d = new Float32Array(h * w);
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            d[y * w + x] = (x + y) / (h + w);
        }
    }
    return d;
}

function rgbInterleaved(h: number, w: number): Uint8Array {
    const d = new Uint8Array(h * w * 3);
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const i = (y * w + x) * 3;
            d[i] = Math.round((x / w) * 255);
            d[i + 1] = Math.round((y / h) * 255);
            d[i + 2] = 128;
        }
    }
    return d;
}

function rgbPlanar(h: number, w: number): Float32Array {
    const d = new Float32Array(3 * h * w);
    const plane = h * w;
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const p = y * w + x;
            d[p] = x / w;
            d[plane + p] = y / h;
            d[2 * plane + p] = 0.5;
        }
    }
    return d;
}

function batchNchw(n: number, h: number, w: number): Float32Array {
    const d = new Float32Array(n * 3 * h * w);
    const plane = h * w;
    for (let b = 0; b < n; b++) {
        const base = b * 3 * plane;
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const p = y * w + x;
                d[base + p] = ((x / w) * (b + 1)) / n;
                d[base + plane + p] = y / h;
                d[base + 2 * plane + p] = b / n;
            }
        }
    }
    return d;
}

function batchNhwc(n: number, h: number, w: number): Uint8Array {
    const d = new Uint8Array(n * h * w * 3);
    for (let b = 0; b < n; b++) {
        const base = b * h * w * 3;
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const i = base + (y * w + x) * 3;
                d[i] = b === 0 ? Math.round((x / w) * 255) : 255 - Math.round((x / w) * 255);
                d[i + 1] = Math.round((y / h) * 255);
                d[i + 2] = b === 0 ? 64 : 192;
            }
        }
    }
    return d;
}

export const SAMPLES: TensorView[] = [
    view('scalar', [1], 'float32', new Float32Array([3.14159])),
    view('vec8', [8], 'float32', Float32Array.from({ length: 8 }, (_, i) => i * 1.5)),
    view('mat4x5', [4, 5], 'float32', Float32Array.from({ length: 20 }, (_, i) => i - 7.5)),
    view('mat_i64', [2, 3], 'int64', BigInt64Array.from([1n, 2n, 3n, 4n, 5n, 6n])),
    view('gray', [64, 64], 'float32', grayGradient(64, 64)),
    view('rgb_hwc', [48, 64, 3], 'uint8', rgbInterleaved(48, 64)),
    view('rgb_chw', [3, 48, 64], 'float32', rgbPlanar(48, 64)),
    view('batch_nchw', [2, 3, 32, 32], 'float32', batchNchw(2, 32, 32)),
    view('batch_nhwc', [2, 32, 32, 3], 'uint8', batchNhwc(2, 32, 32)),
    view('large_1d', [1024], 'float32', Float32Array.from({ length: 1024 }, (_, i) => Math.sin(i / 16))),
];
