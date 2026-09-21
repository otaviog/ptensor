import type { NumericArray } from 'ptensor-ts';

export interface TensorStats {
    min: number;
    max: number;
    mean: number;
    count: number;
}

/**
 * Scans the whole buffer, so it runs once per tensor and is cached by the
 * caller -- a batched image is tens of millions of elements.
 *
 * The int64 case is split off rather than handled with a per-element
 * `elementAt`: that put a call and an `instanceof` in the hot loop, which at
 * this element count is most of the time spent.
 */
export function computeStats(data: NumericArray): TensorStats {
    if (data.length === 0) {
        return { min: NaN, max: NaN, mean: NaN, count: 0 };
    }
    return data instanceof BigInt64Array ? bigIntStats(data) : numberStats(data);
}

function numberStats(data: Exclude<NumericArray, BigInt64Array>): TensorStats {
    const n = data.length;
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    let sum = 0;
    for (let i = 0; i < n; i++) {
        const v = data[i];
        if (v < min) {
            min = v;
        }
        if (v > max) {
            max = v;
        }
        sum += v;
    }
    return { min, max, mean: sum / n, count: n };
}

function bigIntStats(data: BigInt64Array): TensorStats {
    const n = data.length;
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    let sum = 0;
    for (let i = 0; i < n; i++) {
        const v = Number(data[i]);
        if (v < min) {
            min = v;
        }
        if (v > max) {
            max = v;
        }
        sum += v;
    }
    return { min, max, mean: sum / n, count: n };
}
