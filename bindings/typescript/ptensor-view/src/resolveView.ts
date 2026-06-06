export type ViewMode = 'table' | 'image' | 'large-table';

export interface ImagePlane {
    width: number;
    height: number;
    channels: number;
    layout: 'interleaved' | 'planar';
}

export interface ResolvedView {
    mode: ViewMode;
    image?: ImagePlane;
    /** Number of image tabs (N dim, or 1 for a single image). */
    batch: number;
}

const isChannel = (v: number) => v === 1 || v === 3 || v === 4;

/**
 * Picks a display mode from the shape alone. Small tensors render as a table;
 * 2D/3D/4D shapes with a channel-like dim render as one or more images; the
 * rest fall back to a truncated preview.
 */
export function resolveView(shape: number[], tableThreshold = 256): ResolvedView {
    const total = shape.reduce((a, b) => a * b, 1);

    if (total <= tableThreshold) {
        return { mode: 'table', batch: 1 };
    }

    if (shape.length === 2) {
        const [h, w] = shape;
        return {
            mode: 'image',
            batch: 1,
            image: { height: h, width: w, channels: 1, layout: 'interleaved' },
        };
    }
    if (shape.length === 3) {
        // [H, W, C] or [C, H, W]
        const [a, b, c] = shape;
        if (isChannel(c)) {
            return {
                mode: 'image',
                batch: 1,
                image: { height: a, width: b, channels: c, layout: 'interleaved' },
            };
        }
        if (isChannel(a)) {
            return {
                mode: 'image',
                batch: 1,
                image: { height: b, width: c, channels: a, layout: 'planar' },
            };
        }
    }
    if (shape.length === 4) {
        // [N, C, H, W] or [N, H, W, C]
        const [n, a, b, c] = shape;
        if (isChannel(a)) {
            return {
                mode: 'image',
                batch: n,
                image: { height: b, width: c, channels: a, layout: 'planar' },
            };
        }
        if (isChannel(c)) {
            return {
                mode: 'image',
                batch: n,
                image: { height: a, width: b, channels: c, layout: 'interleaved' },
            };
        }
    }
    return { mode: 'large-table', batch: 1 };
}
