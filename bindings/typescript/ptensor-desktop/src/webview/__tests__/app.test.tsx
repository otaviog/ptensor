import { describe, expect, mock, test } from 'bun:test';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import type { Tensor } from '@ptensor/tensor-view';
import type { ServerInfo, TensorPayload } from '../../shared/rpc';

// The real TensorViewer is stubbed: `@ptensor/tensor-view` is a `file:` link
// with its own React copy, and only the bundler dedupes those (see
// src/build/dedupeReact.ts) — under `bun test` two Reacts would collide. What
// is under test here is App's own wiring: session list, follow mode, selection.
// `tensorFromJson` is stubbed with it, so no base64 is decoded here either.
mock.module('@ptensor/tensor-view', () => ({
    TensorViewer: ({ name }: { tensor: Tensor; name?: string }) => (
        <div className="ptv-root">{name}</div>
    ),
    tensorFromJson: (): Tensor => ({
        dtype: 'float32',
        shape: [2, 2],
        stride: [2, 1],
        data: new Float32Array([0, 0.25, 0.5, 1]),
    }),
}));

const { App } = await import('../App');

const INFO: ServerInfo = {
    host: '127.0.0.1',
    port: 8791,
    listening: true,
    maxTensorsPerSession: 100,
};

function payload(sessionId: string, name: string): TensorPayload {
    return {
        sessionId,
        name,
        receivedAt: Date.now(),
        tensor: {
            dtype: 'float32',
            shape: [2, 2],
            stride: [2, 1],
            size_bytes: 16,
            encoding: 'base64',
            blob: 'AAAAAAAAAAAAAAAAAAAAAA==',
        },
    };
}

/** Lets the pending getServerInfo promise settle inside act(). */
async function flush(): Promise<void> {
    await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
    });
}

function mount() {
    const container = document.createElement('div');
    document.body.appendChild(container);
    return { container, root: createRoot(container) };
}

/**
 * Renders App and returns the push function it subscribed with, so a test can
 * feed the panel the way the bun process does.
 */
async function render(info: ServerInfo | Promise<never> = INFO) {
    const { container, root } = mount();
    let push: (payload: TensorPayload) => void = () => {};
    await act(async () => {
        root.render(
            <App
                subscribe={(onTensor) => {
                    push = onTensor;
                }}
                getServerInfo={() => (info instanceof Promise ? info : Promise.resolve(info))}
            />
        );
    });
    await flush();
    return {
        container,
        async push(...payloads: TensorPayload[]) {
            await act(async () => {
                for (const item of payloads) {
                    push(item);
                }
            });
        },
        async unmount() {
            await act(async () => root.unmount());
            container.remove();
        },
    };
}

describe('App', () => {
    test('an empty feed shows the hint instead of a panel', async () => {
        const view = await render({ ...INFO, listening: false, error: 'address in use' });

        expect(view.container.textContent).toContain('No tensors yet');
        expect(view.container.textContent).toContain('feed down: address in use');
        expect(view.container.querySelector('.ptv-root')).toBeNull();

        await view.unmount();
    });

    test('shows a pushed tensor and follows the newest one', async () => {
        const view = await render();
        await view.push(payload('run-a', 'first'));

        expect(view.container.textContent).toContain('run-a');
        expect(view.container.textContent).toContain('listening on 127.0.0.1:8791');
        expect(view.container.querySelector('.ptv-root')?.textContent).toBe('first');

        await view.push(payload('run-a', 'second'));

        // Follow mode is on, so the panel moved to the newest tensor.
        expect(view.container.querySelector('.ptv-root')?.textContent).toBe('second');
        expect(view.container.textContent).toContain('2 kept / 2 received');

        await view.unmount();
    });

    test('groups tensors by the session they were pushed under', async () => {
        const view = await render();
        await view.push(payload('run-a', 'first'), payload('run-b', 'other'));

        const ids = [...view.container.querySelectorAll('.session-id')].map((n) => n.textContent);
        expect(ids.sort()).toEqual(['run-a', 'run-b']);
        // The newest session is listed first and its tensor is on screen.
        expect(ids[0]).toBe('run-a');
        expect(view.container.querySelector('.ptv-root')?.textContent).toBe('other');

        await view.unmount();
    });

    test('clicking an older tensor pins it, and later pushes leave it alone', async () => {
        const view = await render();
        await view.push(payload('run-a', 'first'), payload('run-a', 'second'));

        // Newest first in the list, so the second entry is the older tensor.
        const items = [...view.container.querySelectorAll('.tensor-item')];
        await act(async () => {
            (items[1] as HTMLButtonElement).click();
        });
        expect(view.container.querySelector('.ptv-root')?.textContent).toBe('first');

        const follow = view.container.querySelector('.follow input') as HTMLInputElement;
        await act(async () => {
            follow.click();
        });
        await view.push(payload('run-a', 'third'));

        expect(view.container.querySelector('.ptv-root')?.textContent).toBe('first');

        await view.unmount();
    });

    test('clears one session and then all of them', async () => {
        const view = await render();
        await view.push(payload('run-a', 'first'), payload('run-b', 'other'));

        const clearRunA = [...view.container.querySelectorAll('.session')]
            .find((node) => node.querySelector('.session-id')?.textContent === 'run-a')
            ?.querySelector('.session-clear') as HTMLButtonElement;
        await act(async () => {
            clearRunA.click();
        });

        expect(view.container.textContent).not.toContain('run-a');
        expect(view.container.textContent).toContain('run-b');

        await act(async () => {
            (view.container.querySelector('.clear-all') as HTMLButtonElement).click();
        });

        expect(view.container.textContent).toContain('No tensors yet');
        expect(view.container.querySelector('.ptv-root')).toBeNull();

        await view.unmount();
    });

    test('keeps the cap the bun process reports, trimming what already arrived', async () => {
        const { container, root } = mount();
        let push: (payload: TensorPayload) => void = () => {};
        // Held open so the first pushes land before the cap is known.
        let answer: (info: ServerInfo) => void = () => {};
        const info = new Promise<ServerInfo>((resolve) => {
            answer = resolve;
        });

        await act(async () => {
            root.render(
                <App subscribe={(onTensor) => { push = onTensor; }} getServerInfo={() => info} />
            );
        });
        await act(async () => {
            for (const name of ['one', 'two', 'three']) {
                push(payload('run-a', name));
            }
        });
        expect(container.textContent).toContain('3 kept / 3 received');

        await act(async () => {
            answer({ ...INFO, maxTensorsPerSession: 2 });
        });
        await flush();

        const names = [...container.querySelectorAll('.tensor-name')].map((n) => n.textContent);
        expect(names).toEqual(['three', 'two']);
        expect(container.textContent).toContain('2 kept / 3 received');

        await act(async () => root.unmount());
        container.remove();
    });

    test('drops the oldest tensors past the history cap', async () => {
        const { container, root } = mount();
        let push: (payload: TensorPayload) => void = () => {};
        await act(async () => {
            root.render(
                <App
                    subscribe={(onTensor) => {
                        push = onTensor;
                    }}
                    getServerInfo={() => Promise.resolve(INFO)}
                    maxTensorsPerSession={2}
                />
            );
        });
        await act(async () => {
            for (const name of ['one', 'two', 'three']) {
                push(payload('run-a', name));
            }
        });

        const names = [...container.querySelectorAll('.tensor-name')].map((n) => n.textContent);
        expect(names).toEqual(['three', 'two']);
        expect(container.textContent).toContain('2 kept / 3 received');

        await act(async () => root.unmount());
        container.remove();
    });
});
