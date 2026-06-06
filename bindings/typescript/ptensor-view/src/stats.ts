import { elementAt, type NumericArray } from './types';

export interface TensorStats {
    min: number;
    max: number;
    mean: number;
    count: number;
}

export function computeStats(data: NumericArray): TensorStats {
    const n = data.length;
    if (n === 0) {
        return { min: NaN, max: NaN, mean: NaN, count: 0 };
    }
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    let sum = 0;
    for (let i = 0; i < n; i++) {
        const v = elementAt(data, i);
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
