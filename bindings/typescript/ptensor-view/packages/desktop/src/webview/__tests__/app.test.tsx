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
    TensorViewer: ({ tensor, name }: { tensor: Tensor; name?: string }) => (
        <div className="ptv-root">
            {name}
            {/* Which arrival is on screen. Every arrival of one name shares the
                name, and the decoded-tensor cache means a re-selection reads
                nothing, so the id the fake feed tagged the tensor with is the
                only way to see what the panel is actually showing. */}
            <span className="shown-id">{(tensor as TaggedTensor).id}</span>
        </div>
    ),
}));

// Imported after the mock is installed, so App picks up the stub.
const { App } = await import('../App');
const { TensorCache } = await import('../tensorCache');

const INFO: FeedInfo = { host: '127.0.0.1', port: 8791, listening: true };

/** A decoded tensor, tagged with the id it was read for. */
type TaggedTensor = Tensor & { id?: string };

function tensorFor(id: string): TaggedTensor {
    return {
        dtype: 'float32',
        shape: [2, 2],
        stride: [2, 1],
        data: new Float32Array([0, 0.25, 0.5, 1]),
        id,
    };
}

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
            return tensorFor(id);
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
        /**
         * Drags one slider to `index`. React tracks the input's value itself,
         * so the write has to go through the native setter for the change to
         * be seen.
         */
        async scrub(slider: HTMLInputElement, index: number) {
            const setValue = Object.getOwnPropertyDescriptor(
                HTMLInputElement.prototype,
                'value'
            )?.set;
            await act(async () => {
                setValue?.call(slider, String(index));
                slider.dispatchEvent(new Event('input', { bubbles: true }));
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
        await view.emit(hello([]));
        await view.emit(arrived(meta('frame', 'frame')));

        expect(view.container.textContent).toContain('run-a');
        expect(view.container.textContent).toContain('listening on 127.0.0.1:8791');
        expect(shownId(view)).toBe('frame');
        expect(view.loads).toEqual(['frame']);

        await view.emit(arrived(meta('frame-2', 'frame')));

        // Follow mode is on, so the panel moved to the newest arrival.
        expect(shownId(view)).toBe('frame-2');
        expect(view.loads).toEqual(['frame', 'frame-2']);
        expect(view.container.textContent).toContain('2 held');

        await view.unmount();
    });

    test('lists one row per name, however many times it arrived', async () => {
        const view = await render();
        await view.emit(hello([]));
        for (const id of ['frame', 'frame-2', 'frame-3']) {
            await view.emit(arrived(meta(id, 'frame')));
        }
        await view.emit(arrived(meta('weights', 'weights')));

        // Four arrivals, two names.
        expect(names(view)).toEqual(['frame', 'weights']);
        expect(view.container.textContent).toContain('4 held');

        await view.unmount();
    });

    test('keeps the names in the order they first appeared', async () => {
        const view = await render();
        await view.emit(hello([]));
        await view.emit(arrived(meta('a1', 'alpha')));
        await view.emit(arrived(meta('b1', 'beta')));
        // Alpha arrives again: a recency order would jump it back to the top,
        // which makes the sidebar unusable on an alternating feed.
        await view.emit(arrived(meta('a2', 'alpha')));

        expect(names(view)).toEqual(['alpha', 'beta']);

        await view.unmount();
    });

    test('offers a slider only once a name has arrived more than once', async () => {
        const view = await render();
        await view.emit(hello([]));
        await view.emit(arrived(meta('frame', 'frame')));

        expect(sliders(view)).toHaveLength(0);

        await view.emit(arrived(meta('frame-2', 'frame')));

        const [slider] = sliders(view);
        expect(slider.max).toBe('1');
        expect(slider.value).toBe('1');
        expect(view.container.textContent).toContain('2/2');

        await view.unmount();
    });

    test('scrubbing shows an older arrival and stops following', async () => {
        const view = await render();
        await view.emit(hello([]));
        for (const id of ['frame', 'frame-2', 'frame-3']) {
            await view.emit(arrived(meta(id, 'frame')));
        }
        expect(view.loads).toEqual(['frame', 'frame-2', 'frame-3']);

        await view.scrub(sliders(view)[0], 0);

        expect(shownId(view)).toBe('frame');
        expect(view.container.textContent).toContain('1/3');
        // Scrubbing off the newest turns follow off, so the next arrival cannot
        // yank the panel back to the end.
        expect(followBox(view).checked).toBe(false);

        await view.emit(arrived(meta('frame-4', 'frame')));

        // Still on the arrival that was pinned, and it was never re-read.
        expect(shownId(view)).toBe('frame');
        expect(view.container.textContent).toContain('1/4');
        expect(view.loads).toEqual(['frame', 'frame-2', 'frame-3']);

        await view.unmount();
    });

    test('scrubbing back to the newest leaves follow off until asked', async () => {
        const view = await render();
        await view.emit(hello([]));
        await view.emit(arrived(meta('frame', 'frame')));
        await view.emit(arrived(meta('frame-2', 'frame')));

        await view.scrub(sliders(view)[0], 0);
        expect(followBox(view).checked).toBe(false);

        await view.scrub(sliders(view)[0], 1);

        // Back at the end, but following is the checkbox's business: nothing
        // should re-arm it behind the user's back.
        expect(view.container.textContent).toContain('2/2');
        expect(followBox(view).checked).toBe(false);

        await view.unmount();
    });

    test('each name keeps its own slider position', async () => {
        const view = await render();
        await view.emit(hello([]));
        for (const id of ['a1', 'a2', 'a3']) {
            await view.emit(arrived(meta(id, 'alpha')));
        }
        for (const id of ['b1', 'b2', 'b3']) {
            await view.emit(arrived(meta(id, 'beta')));
        }

        await view.scrub(sliders(view)[0], 0); // alpha -> first
        await view.scrub(sliders(view)[1], 1); // beta  -> second

        const shown = [...view.container.querySelectorAll('.tensor-pos')].map((n) => n.textContent);
        expect(shown).toEqual(['1/3', '2/3']);

        // Switching rows shows that row's position, not a shared one.
        await view.click(view.container.querySelectorAll('.tensor-item')[0]);
        expect(shownId(view)).toBe('a1');
        await view.click(view.container.querySelectorAll('.tensor-item')[1]);
        expect(shownId(view)).toBe('b2');

        await view.unmount();
    });

    test('catches up on what the store already holds when it connects', async () => {
        const view = await render();
        // Newest first, the order the store lists them in.
        await view.emit(
            hello([meta('a3', 'alpha'), meta('b1', 'beta'), meta('a2', 'alpha'), meta('a1', 'alpha')])
        );

        expect(names(view)).toEqual(['alpha', 'beta']);
        expect(sliders(view)[0].max).toBe('2');
        // Nothing is on screen until something is selected, so nothing is read.
        expect(view.loads).toEqual([]);

        await view.unmount();
    });

    test('groups tensors by the session they were pushed under', async () => {
        const view = await render();
        await view.emit(hello([]));
        await view.emit(arrived(meta('first', 'first', 'run-a')));
        await view.emit(arrived(meta('other', 'other', 'run-b')));

        const ids = [...view.container.querySelectorAll('.session-id')].map((n) => n.textContent);
        expect(ids.sort()).toEqual(['run-a', 'run-b']);
        // The newest session is listed first and its tensor is on screen.
        expect(ids[0]).toBe('run-a');
        expect(shownId(view)).toBe('other');

        await view.unmount();
    });

    test('reads a tensor once and then takes it from the cache', async () => {
        const view = await render({ cache: new TensorCache() });
        await view.emit(hello([]));
        await view.emit(arrived(meta('frame', 'frame')));
        await view.emit(arrived(meta('frame-2', 'frame')));
        expect(view.loads).toEqual(['frame', 'frame-2']);

        await view.scrub(sliders(view)[0], 0);
        await view.scrub(sliders(view)[0], 1);

        // Both were decoded already, so scrubbing over them reads nothing.
        expect(view.loads).toEqual(['frame', 'frame-2']);

        await view.unmount();
    });

    test('drops what the store evicted, so the slider matches what is held', async () => {
        const view = await render();
        await view.emit(hello([]));
        for (const id of ['frame', 'frame-2', 'frame-3']) {
            await view.emit(arrived(meta(id, 'frame')));
        }
        expect(sliders(view)[0].max).toBe('2');

        await view.emit(arrived(meta('frame-4', 'frame'), ['frame']));

        // Three arrivals left, so three slider stops.
        expect(sliders(view)[0].max).toBe('2');
        expect(view.container.textContent).toContain('3/3');
        expect(view.container.textContent).toContain('3 held');

        await view.unmount();
    });

    test('drops a name whose every arrival was evicted', async () => {
        const view = await render();
        await view.emit(hello([]));
        await view.emit(arrived(meta('a1', 'alpha')));
        await view.emit(arrived(meta('b1', 'beta'), ['a1']));

        expect(names(view)).toEqual(['beta']);

        await view.unmount();
    });

    test('says so when a tensor cannot be read', async () => {
        const view = await render({ failLoad: 'the tensor request answered 404' });
        await view.emit(hello([]));
        await view.emit(arrived(meta('frame', 'frame')));

        expect(view.container.textContent).toContain('the tensor request answered 404');
        expect(view.container.querySelector('.ptv-root')).toBeNull();

        await view.unmount();
    });

    test('clears one session and then all of them', async () => {
        const view = await render();
        await view.emit(hello([]));
        await view.emit(arrived(meta('first', 'first', 'run-a')));
        await view.emit(arrived(meta('other', 'other', 'run-b')));

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

/** The names the sidebar lists, in order. */
function names(view: { container: HTMLElement }): (string | null)[] {
    return [...view.container.querySelectorAll('.tensor-name')].map((n) => n.textContent);
}

/** The time sliders, one per name that arrived more than once. */
function sliders(view: { container: HTMLElement }): HTMLInputElement[] {
    return [...view.container.querySelectorAll('.tensor-slider')] as HTMLInputElement[];
}

function followBox(view: { container: HTMLElement }): HTMLInputElement {
    return view.container.querySelector('.follow input') as HTMLInputElement;
}

/** The id of the arrival the panel is showing. */
function shownId(view: { container: HTMLElement }): string | null | undefined {
    return view.container.querySelector('.shown-id')?.textContent;
}
