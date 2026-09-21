import { describe, expect, test } from 'bun:test';
import type { Tensor } from '@ptensor/tensor-view';
import { TensorCache } from '../tensorCache';

/** A tensor of exactly `bytes` decoded bytes. */
function tensor(bytes: number): Tensor {
    const elements = bytes / 4;
    return {
        dtype: 'float32',
        shape: [elements],
        stride: [1],
        data: new Float32Array(elements),
    };
}

describe('TensorCache', () => {
    test('hands back what it was given and counts the decoded bytes', () => {
        const cache = new TensorCache(1024);
        const held = tensor(400);
        cache.put('a', held);

        expect(cache.get('a')).toBe(held);
        expect(cache.usedBytes).toBe(400);
        expect(cache.get('missing')).toBeUndefined();
    });

    test('drops the least recently used tensor to stay under the budget', () => {
        const cache = new TensorCache(1000);
        cache.put('a', tensor(400));
        cache.put('b', tensor(400));

        // Reading 'a' makes 'b' the one that goes.
        expect(cache.get('a')).toBeDefined();
        cache.put('c', tensor(400));

        expect(cache.get('b')).toBeUndefined();
        expect(cache.get('a')).toBeDefined();
        expect(cache.get('c')).toBeDefined();
        expect(cache.usedBytes).toBe(800);
    });

    test('keeps the newest tensor even when it is alone over the budget', () => {
        const cache = new TensorCache(100);
        cache.put('a', tensor(400));

        // It is the one on screen; dropping it would leave nothing to draw.
        expect(cache.get('a')).toBeDefined();
        expect(cache.size).toBe(1);
    });

    test('re-putting the same id replaces it rather than double counting', () => {
        const cache = new TensorCache(10_000);
        cache.put('a', tensor(400));
        cache.put('a', tensor(800));

        expect(cache.usedBytes).toBe(800);
        expect(cache.size).toBe(1);
    });

    test('forgets a tensor on request, and all of them', () => {
        const cache = new TensorCache(10_000);
        cache.put('a', tensor(400));
        cache.put('b', tensor(400));

        cache.delete('a');
        expect(cache.get('a')).toBeUndefined();
        expect(cache.usedBytes).toBe(400);

        cache.clear();
        expect(cache.usedBytes).toBe(0);
        expect(cache.size).toBe(0);
    });
});
