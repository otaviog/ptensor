import { describe, expect, test } from 'bun:test';
import { imageMapping, planeToRgba } from '../imageData';
import type { ImagePlane } from '../resolveView';

const IDENTITY = { lo: 0, scale: 1 };

function ramp(length: number): Uint8Array {
    const out = new Uint8Array(length);
    for (let i = 0; i < length; i++) {
        out[i] = (i * 7) & 0xff;
    }
    return out;
}

describe('planeToRgba', () => {
    test('uint8 maps straight through, with an opaque alpha', () => {
        const plane: ImagePlane = { width: 2, height: 1, channels: 3, layout: 'interleaved' };
        const rgba = planeToRgba(new Uint8Array([1, 2, 3, 4, 5, 6]), 0, plane, IDENTITY);
        expect(Array.from(rgba)).toEqual([1, 2, 3, 255, 4, 5, 6, 255]);
    });

    test('a single channel is written to r, g and b', () => {
        const plane: ImagePlane = { width: 2, height: 1, channels: 1, layout: 'interleaved' };
        const rgba = planeToRgba(new Uint8Array([9, 10]), 0, plane, IDENTITY);
        expect(Array.from(rgba)).toEqual([9, 9, 9, 255, 10, 10, 10, 255]);
    });

    test('four channels keep their own alpha', () => {
        const plane: ImagePlane = { width: 1, height: 1, channels: 4, layout: 'interleaved' };
        const rgba = planeToRgba(new Uint8Array([1, 2, 3, 4]), 0, plane, IDENTITY);
        expect(Array.from(rgba)).toEqual([1, 2, 3, 4]);
    });

    test('offset selects the plane of a batch', () => {
        const plane: ImagePlane = { width: 1, height: 1, channels: 3, layout: 'interleaved' };
        const batch = new Uint8Array([1, 2, 3, 7, 8, 9]);
        expect(Array.from(planeToRgba(batch, 3, plane, IDENTITY))).toEqual([7, 8, 9, 255]);
    });

    // The uint8 fast path and the generic path must agree: the first is only an
    // optimization of the second for identity-mapped interleaved bytes.
    test('the uint8 fast path matches the generic path', () => {
        for (const channels of [1, 3, 4] as const) {
            const plane: ImagePlane = { width: 8, height: 5, channels, layout: 'interleaved' };
            const data = ramp(plane.width * plane.height * channels);
            const fast = planeToRgba(data, 0, plane, IDENTITY);
            // Float32Array of the same values never takes the byte fast path.
            const generic = planeToRgba(Float32Array.from(data), 0, plane, IDENTITY);
            expect(Array.from(fast)).toEqual(Array.from(generic));
        }
    });

    test('planar and interleaved address the same pixels differently', () => {
        const size = { width: 2, height: 1, channels: 3 } as const;
        const interleaved = planeToRgba(
            new Uint8Array([1, 2, 3, 4, 5, 6]),
            0,
            { ...size, layout: 'interleaved' },
            IDENTITY
        );
        const planar = planeToRgba(
            new Uint8Array([1, 4, 2, 5, 3, 6]),
            0,
            { ...size, layout: 'planar' },
            IDENTITY
        );
        expect(Array.from(planar)).toEqual(Array.from(interleaved));
    });

    test('a float plane is stretched by the tensor range', () => {
        const plane: ImagePlane = { width: 2, height: 1, channels: 1, layout: 'interleaved' };
        const mapping = imageMapping('float32', { min: 0, max: 2, mean: 1, count: 2 });
        const rgba = planeToRgba(Float32Array.from([0, 2]), 0, plane, mapping);
        expect(Array.from(rgba)).toEqual([0, 0, 0, 255, 255, 255, 255, 255]);
    });
});
