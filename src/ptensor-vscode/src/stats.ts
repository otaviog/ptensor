export interface TensorStats {
    min: number;
    max: number;
    mean: number;
    count: number;
}

export function computeStats(data: ArrayLike<number> | BigInt64Array): TensorStats {
    const n = data.length;
    if (n === 0) {
        return { min: NaN, max: NaN, mean: NaN, count: 0 };
    }
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    let sum = 0;
    if (data instanceof BigInt64Array) {
        for (let i = 0; i < n; i++) {
            const v = Number(data[i]);
            if (v < min) { min = v; }
            if (v > max) { max = v; }
            sum += v;
        }
    } else {
        for (let i = 0; i < n; i++) {
            const v = (data as ArrayLike<number>)[i];
            if (v < min) { min = v; }
            if (v > max) { max = v; }
            sum += v;
        }
    }
    return { min, max, mean: sum / n, count: n };
}
