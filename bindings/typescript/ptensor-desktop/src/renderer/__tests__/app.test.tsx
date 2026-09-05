import { describe, expect, mock, test } from 'bun:test';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import type { TensorView } from '@ptensor/tensor-view';
import type { ServerInfo, SessionSummary } from '../../shared/protocol';

// The real TensorViewer is stubbed: `@ptensor/tensor-view` is a `file:` link
// with its own React copy, and only the bundler dedupes those (see
// src/build/dedupeReact.ts) — under `bun test` two Reacts would collide. What
// is under test here is App's own wiring: session list, follow mode, selection.
mock.module('@ptensor/tensor-view', () => ({
    TensorViewer: ({ tensor }: { tensor: TensorView }) => (
        <div className="ptv-root">{tensor.name}</div>
    ),
}));

const { App } = await import('../App');

function tensorView(name: string): TensorView {
    return {
        name,
        dtype: 'float32',
        shape: [2n, 2n],
        stride: [2n, 1n],
        array: new Float32Array([0, 0.25, 0.5, 1]),
    };
}

function sessionList(): SessionSummary[] {
    const now = Date.now();
    return [
        {
            id: 'run-a',
            updatedAt: now,
            totalReceived: 2,
            tensors: [
                {
                    id: 't1',
                    name: 'first',
                    receivedAt: now,
                    dtype: 'float32',
                    shape: [2, 2],
                    elems: 4,
                    bytes: 16,
                },
                {
                    id: 't2',
                    name: 'second',
                    receivedAt: now,
                    dtype: 'float32',
                    shape: [2, 2],
                    elems: 4,
                    bytes: 16,
                },
            ],
        },
    ];
}

/** Lets the pending loadTensor promise settle inside act(). */
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

describe('App', () => {
    test('lists sessions and shows the newest tensor of the active one', async () => {
        const { container, root } = mount();
        const info: ServerInfo = { host: '127.0.0.1', port: 8791, listening: true };
        const requested: string[] = [];

        await act(async () => {
            root.render(
                <App
                    loadTensor={async (_sessionId, tensorId) => {
                        requested.push(tensorId);
                        return tensorView(tensorId === 't2' ? 'second' : 'first');
                    }}
                    clearSession={() => {}}
                    subscribe={(onSessions, onInfo) => {
                        onSessions(sessionList());
                        onInfo(info);
                    }}
                />
            );
        });

        await flush();

        expect(container.textContent).toContain('run-a');
        expect(container.textContent).toContain('listening on 127.0.0.1:8791');
        // Follow mode selects the newest tensor, and its panel renders.
        expect(requested).toEqual(['t2']);
        expect(container.querySelector('.ptv-root')?.textContent).toBe('second');
        expect(container.textContent).toContain('2 kept / 2 received');

        await act(async () => root.unmount());
        container.remove();
    });

    test('clicking an older tensor loads that one instead', async () => {
        const { container, root } = mount();
        const requested: string[] = [];

        await act(async () => {
            root.render(
                <App
                    loadTensor={async (_sessionId, tensorId) => {
                        requested.push(tensorId);
                        return tensorView(tensorId);
                    }}
                    clearSession={() => {}}
                    subscribe={(onSessions) => onSessions(sessionList())}
                />
            );
        });

        // Newest first in the list, so the second entry is the older tensor.
        const items = [...container.querySelectorAll('.tensor-item')];
        await act(async () => {
            (items[1] as HTMLButtonElement).click();
        });

        await flush();

        expect(requested).toEqual(['t2', 't1']);
        expect(container.querySelector('.ptv-root')?.textContent).toBe('t1');

        await act(async () => root.unmount());
        container.remove();
    });

    test('an empty feed shows the hint instead of a panel', async () => {
        const { container, root } = mount();
        await act(async () => {
            root.render(
                <App
                    loadTensor={async () => null}
                    clearSession={() => {}}
                    subscribe={(onSessions, onInfo) => {
                        onSessions([]);
                        onInfo({
                            host: '127.0.0.1',
                            port: 8791,
                            listening: false,
                            error: 'address in use',
                        });
                    }}
                />
            );
        });

        await flush();

        expect(container.textContent).toContain('No tensors yet');
        expect(container.textContent).toContain('feed down: address in use');
        expect(container.querySelector('.ptv-root')).toBeNull();

        await act(async () => root.unmount());
        container.remove();
    });
});
