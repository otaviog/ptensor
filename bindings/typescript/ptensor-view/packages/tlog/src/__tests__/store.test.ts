import { describe, expect, test } from 'bun:test';
import type { TensorJson } from 'ptensor-ts';
import type { TensorPayload } from '../connectionHandler';
import { TensorStore } from '../store';

function tensor(sizeBytes = 4): TensorJson {
    return {
        dtype: 'float32',
        shape: [sizeBytes / 4],
        stride: [1],
        size_bytes: sizeBytes,
        encoding: 'base64',
        blob: 'A'.repeat(Math.ceil(sizeBytes / 3) * 4),
    };
}

function payload(name: string, sessionId = 'run-a', size = 4): TensorPayload {
    return { sessionId, name, tensor: tensor(size), receivedAt: Date.now() };
}

describe('TensorStore', () => {
    test('serves back the JSON text it was given, verbatim', () => {
        const store = new TensorStore();
        const entry = payload('frame');
        const { meta } = store.add(entry);

        expect(store.json(meta.id)).toBe(JSON.stringify(entry.tensor));
        expect(JSON.parse(store.json(meta.id)!)).toEqual(entry.tensor);
    });

    test('lifts the metadata out of the tensor and leaves the blob behind', () => {
        const store = new TensorStore();
        const { meta } = store.add(payload('frame', 'run-a', 16));

        expect(meta).toMatchObject({
            id: 'frame',
            sessionId: 'run-a',
            name: 'frame',
            dtype: 'float32',
            size_bytes: 16,
            encoding: 'base64',
        });
        expect(meta).not.toHaveProperty('blob');
        expect(meta.storedBytes).toBe(store.usedBytes);
    });

    test('gives repeated names distinct, URL-safe ids', () => {
        const store = new TensorStore();

        expect(store.add(payload('frame')).meta.id).toBe('frame');
        expect(store.add(payload('frame')).meta.id).toBe('frame-2');
        expect(store.add(payload('frame')).meta.id).toBe('frame-3');
        // Whatever a producer labels a tensor has to survive being a path segment.
        expect(store.add(payload('layer 3 / relu out')).meta.id).toBe('layer-3-relu-out');
        expect(store.add(payload('')).meta.id).toBe('tensor');
    });

    test('never hands a freed id to a later tensor', () => {
        const store = new TensorStore({ budgetBytes: 1 });
        const { meta: first } = store.add(payload('frame'));
        const { meta: second } = store.add(payload('frame'));

        // The first was evicted, so its id resolves to nothing -- not to the
        // tensor that came after it.
        expect(store.json(first.id)).toBeUndefined();
        expect(second.id).not.toBe(first.id);
    });

    test('lists newest first', () => {
        const store = new TensorStore();
        store.add(payload('one'));
        store.add(payload('two'));
        store.add(payload('three'));

        expect(store.list().map((meta) => meta.name)).toEqual(['three', 'two', 'one']);
    });

    test('evicts the oldest tensors to stay under the byte budget', () => {
        // Sized off what one tensor actually costs, so the budget holds two of
        // them and no more.
        const each = JSON.stringify(tensor(300)).length;
        const budgetBytes = each * 2 + 1;
        const evicted: string[] = [];
        const store = new TensorStore({
            budgetBytes,
            logger: { info: (message) => evicted.push(message) },
        });

        const { meta: first } = store.add(payload('one', 'run-a', 300));
        const { meta: second } = store.add(payload('two', 'run-a', 300));
        expect(store.list()).toHaveLength(2);

        const third = store.add(payload('three', 'run-a', 300));

        expect(store.json(first.id)).toBeUndefined();
        expect(store.json(second.id)).toBeDefined();
        expect(store.json(third.meta.id)).toBeDefined();
        expect(store.usedBytes).toBeLessThanOrEqual(budgetBytes);
        expect(evicted.join(' ')).toContain(first.id);
        // The add reports what it dropped, so a mirror of the store can follow.
        expect(third.evicted.map((meta) => meta.id)).toEqual([first.id]);
    });

    test('keeps a tensor that is over the budget on its own', () => {
        const warned: string[] = [];
        const store = new TensorStore({
            budgetBytes: 10,
            logger: { warn: (message) => warned.push(message) },
        });

        const { meta } = store.add(payload('huge', 'run-a', 4096));

        // Dropping the tensor the user just asked to see would be worse than
        // going over for one of them.
        expect(store.json(meta.id)).toBeDefined();
        expect(warned.join(' ')).toContain('over the');
    });

    test('clears one session, then everything', () => {
        const store = new TensorStore();
        const { meta: a } = store.add(payload('one', 'run-a'));
        const { meta: b } = store.add(payload('two', 'run-b'));

        store.clear('run-a');
        expect(store.json(a.id)).toBeUndefined();
        expect(store.json(b.id)).toBeDefined();
        expect(store.usedBytes).toBe(b.storedBytes);

        store.clear();
        expect(store.list()).toEqual([]);
        expect(store.usedBytes).toBe(0);
    });
});
