export function formatNumber(v: number): string {
    if (!Number.isFinite(v)) {
        return String(v);
    }
    if (Number.isInteger(v)) {
        return v.toString();
    }
    return v.toPrecision(6);
}
