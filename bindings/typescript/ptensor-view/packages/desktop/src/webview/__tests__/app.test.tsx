import { describe, expect, mock, test } from 'bun:test';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import type { Tensor } from '@ptensor/tensor-view';
import type { FeedEvent, FeedInfo, TensorMeta } from '../../shared/feed';
import type { FeedClient } from '../feedClient';

// The real TensorViewer is stubbed: `@ptensor/tensor-view` is a `file:` link
// with its own React copy, and only the bundler dedupes those (see
// src/build/dedupeReact.ts) — under `bun test` two Reacts would collide. What
// is under test here is App's own wiring: session list, follow mode, selection,
// and what it does with the feed's events.
//
// `mock.module` replaces the module for the whole test process, so the stub has
// to carry everything the other test files import from it at runtime. That is
// only the panel: the decode path reaches ptensor-ts directly.
mock.module('@ptensor/tensor-view', () => ({
    TensorViewer: ({ name }: { tensor: Tensor; name?: string }) => (
        <div className="ptv-root">{name}</div>
    ),
}));

// Imported after the mock is installed, so App picks up the stub.
const { App } = await import('../App');
const { TensorCache } = await import('../tensorCache');

const INFO: FeedInfo = { host: '127.0.0.1', port: 8791, listening: true };

const TENSOR: Tensor = {
    dtype: 'float32',
    shape: [2, 2],
    stride: [2, 1],
    data: new Float32Array([0, 0.25, 0.5, 1]),
};

function meta(id: string, name = id, sessionId = 'run-a'): TensorMeta {
    return {
        id,
        sessionId,
        name,
        dtype: 'float32',
        shape: [2, 2],
        stride: [2, 1],
        size_bytes: 16,
        encoding: 'base64',
        storedBytes: 120,
        receivedAt: Date.now(),
    };
}

/**
 * A feed under the test's control: `emit` plays the socket, and every fetch the
 * panel makes is recorded so a test can tell a re-read from a cache hit.
 */
function fakeFeed(options: { failLoad?: string } = {}) {
    let onEvent: (event: FeedEvent) => void = () => {};
    const loads: string[] = [];
    const cleared: (string | undefined)[] = [];

    const client: FeedClient = {
        subscribe(next) {
            onEvent = next;
            return () => {
                onEvent = () => {};
            };
        },
        listTensors: async () => [],
        loadTensor: async (id) => {
            loads.push(id);
            if (options.failLoad !== undefined) {
                throw new Error(options.failLoad);
            }
            return TENSOR;
        },
        clear: async (sessionId) => {
            cleared.push(sessionId);
            onEvent({ type: 'cleared', sessionId });
        },
    };

    return {
        client,
        loads,
        cleared,
        emit: (event: FeedEvent) => onEvent(event),
    };
}

/** Lets the pending fetches and effects settle inside act(). */
async function flush(): Promise<void> {
    await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
    });
}

async function render(options: { failLoad?: string; cache?: InstanceType<typeof TensorCache> } = {}) {
    const feed = fakeFeed({ failLoad: options.failLoad });
    const container = document.createElement('div');
    document.body.appendChild(container);
    const root = createRoot(container);

    await act(async () => {
        root.render(<App client={feed.client} cache={options.cache} />);
    });
    await flush();

    return {
        container,
        loads: feed.loads,
        cleared: feed.cleared,
        /** Plays one or more feed events, then lets the panel settle. */
        async emit(...events: FeedEvent[]) {
            await act(async () => {
                for (const event of events) {
                    feed.emit(event);
                }
            });
            await flush();
        },
        async click(node: Element) {
            await act(async () => {
                (node as HTMLButtonElement).click();
            });
            await flush();
        },
        async unmount() {
            await act(async () => root.unmount());
            container.remove();
        },
    };
}

function hello(tensors: TensorMeta[], info: FeedInfo = INFO): FeedEvent {
    return { type: 'hello', info, tensors };
}

function arrived(tensor: TensorMeta, dropped: string[] = []): FeedEvent {
    return { type: 'tensor', tensor, dropped };
}

describe('App', () => {
    test('an empty feed shows the hint instead of a panel', async () => {
        const view = await render();
        await view.emit(hello([], { ...INFO, listening: false, error: 'address in use' }));

        expect(view.container.textContent).toContain('No tensors yet');
        expect(view.container.textContent).toContain('feed down: address in use');
        expect(view.container.querySelector('.ptv-root')).toBeNull();

        await view.unmount();
    });

    test('fetches and shows a pushed tensor, and follows the newest', async () => {
        const view = await render();
        await view.emit(hello([]), arrived(meta('first')));

        expect(view.container.textContent).toContain('run-a');
        expect(view.container.textContent).toContain('listening on 127.0.0.1:8791');
        expect(view.container.querySelector('.ptv-root')?.textContent).toBe('first');
        expect(view.loads).toEqual(['first']);

        await view.emit(arrived(meta('second')));

        // Follow mode is on, so the panel moved to the newest tensor.
        expect(view.container.querySelector('.ptv-root')?.textContent).toBe('second');
        expect(view.container.textContent).toContain('2 held');
        expect(view.loads).toEqual(['first', 'second']);

        await view.unmount();
    });

    test('catches up on what the store already holds when it connects', async () => {
        const view = await render();
        await view.emit(hello([meta('third'), meta('second'), meta('first')]));

        // The store lists newest first; the sidebar shows them the same way.
        const names = [...view.container.querySelectorAll('.tensor-name')].map((n) => n.textContent);
        expect(names).toEqual(['third', 'second', 'first']);
        // Nothing is on screen until something is selected, so nothing is read.
        expect(view.loads).toEqual([]);

        await view.unmount();
    });

    test('groups tensors by the session they were pushed under', async () => {
        const view = await render();
        await view.emit(
            hello([]),
            arrived(meta('first', 'first', 'run-a')),
            arrived(meta('other', 'other', 'run-b'))
        );

        const ids = [...view.container.querySelectorAll('.session-id')].map((n) => n.textContent);
        expect(ids.sort()).toEqual(['run-a', 'run-b']);
        // The newest session is listed first and its tensor is on screen.
        expect(ids[0]).toBe('run-a');
        expect(view.container.querySelector('.ptv-root')?.textContent).toBe('other');

        await view.unmount();
    });

    test('clicking an older tensor pins it, and later pushes leave it alone', async () => {
        const view = await render();
        await view.emit(hello([]), arrived(meta('first')), arrived(meta('second')));

        // Newest first in the list, so the second entry is the older tensor.
        const items = [...view.container.querySelectorAll('.tensor-item')];
        await view.click(items[1]);
        expect(view.container.querySelector('.ptv-root')?.textContent).toBe('first');

        await view.click(view.container.querySelector('.follow input')!);
        await view.emit(arrived(meta('third')));

        expect(view.container.querySelector('.ptv-root')?.textContent).toBe('first');

        await view.unmount();
    });

    test('reads a tensor once and then takes it from the cache', async () => {
        const view = await render({ cache: new TensorCache() });
        // One at a time: batched into a single render, only the last one would
        // ever be selected, so only it would be read.
        await view.emit(hello([]));
        await view.emit(arrived(meta('first')));
        await view.emit(arrived(meta('second')));
        expect(view.loads).toEqual(['first', 'second']);

        const items = [...view.container.querySelectorAll('.tensor-item')];
        await view.click(items[1]);
        await view.click(items[0]);

        // Both were decoded already, so going back and forth reads nothing.
        expect(view.loads).toEqual(['first', 'second']);

        await view.unmount();
    });

    test('drops what the store evicted, so the list matches what is held', async () => {
        const view = await render();
        await view.emit(hello([]), arrived(meta('first')), arrived(meta('second')));

        await view.emit(arrived(meta('third'), ['first']));

        const names = [...view.container.querySelectorAll('.tensor-name')].map((n) => n.textContent);
        expect(names).toEqual(['third', 'second']);
        expect(view.container.textContent).toContain('2 held');

        await view.unmount();
    });

    test('says so when a tensor cannot be read', async () => {
        const view = await render({ failLoad: 'tensor first answered 404' });
        await view.emit(hello([]), arrived(meta('first')));

        expect(view.container.textContent).toContain('tensor first answered 404');
        expect(view.container.querySelector('.ptv-root')).toBeNull();

        await view.unmount();
    });

    test('clears one session and then all of them', async () => {
        const view = await render();
        await view.emit(
            hello([]),
            arrived(meta('first', 'first', 'run-a')),
            arrived(meta('other', 'other', 'run-b'))
        );

        const clearRunA = [...view.container.querySelectorAll('.session')]
            .find((node) => node.querySelector('.session-id')?.textContent === 'run-a')
            ?.querySelector('.session-clear');
        await view.click(clearRunA!);

        // The panel does not drop them itself: it asks, and the feed's answer
        // is what updates the list.
        expect(view.cleared).toEqual(['run-a']);
        expect(view.container.textContent).not.toContain('run-a');
        expect(view.container.textContent).toContain('run-b');

        await view.click(view.container.querySelector('.clear-all')!);

        expect(view.cleared).toEqual(['run-a', undefined]);
        expect(view.container.textContent).toContain('No tensors yet');
        expect(view.container.querySelector('.ptv-root')).toBeNull();

        await view.unmount();
    });
});
