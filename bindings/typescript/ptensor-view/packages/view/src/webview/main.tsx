// Webview bootstrap. Built by Vite (lib/iife) into dist/webview.js and loaded
// by the VS Code extension. Mounts the React panel and waits for the host to
// say where a tensor can be read.
//
// The host sends a URL, not the tensor: a `TensorJson` is hundreds of megabytes
// of base64, and `postMessage` into a webview is a JSON stringify, an IPC hop
// and a parse -- several full copies of it, through a channel not built for
// that. The host writes the tensor to a file and hands over a webview URI; this
// side fetches and decodes it, off-thread where it can.

import { StrictMode, useEffect, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { SampleBrowser } from '../components/SampleBrowser';
import { TensorViewer } from '../components/TensorViewer';
import { SAMPLES } from '../samples';
import { createDecoder } from '../decode';
import type { Tensor } from 'ptensor-ts';
// Inlined so the whole webview ships as a single self-contained JS bundle.
import css from '../styles.css?inline';

interface VsCodeApi {
    postMessage(message: unknown): void;
    getState(): unknown;
    setState(state: unknown): void;
}
declare global {
    interface Window {
        acquireVsCodeApi?: () => VsCodeApi;
        /** Initial message embedded in the page by the extension host. */
        __PTENSOR_INIT__?: HostMessage | null;
    }
}

/**
 * Message posted by the extension host: where to read one tensor from. `url` is
 * a webview URI for a file the host wrote, so fetching it stays inside the
 * webview's own resource roots.
 */
interface TensorMessage {
    type: 'tensor';
    name?: string;
    url: string;
    tableThreshold?: number;
    /** Whether the host can re-read this tensor (false for demo/sample tensors). */
    canRefresh?: boolean;
}

/** Demo mode: render the built-in sample tensors, no debugger needed. */
interface DemoMessage {
    type: 'demo';
    tableThreshold?: number;
}

/**
 * The tab is open and the tensor is on its way: the host has asked the debuggee
 * to push it, and it arrives over a socket rather than with this message.
 */
interface PendingMessage {
    type: 'pending';
    name?: string;
}

type HostMessage = TensorMessage | DemoMessage | PendingMessage;

const KINDS = new Set(['tensor', 'demo', 'pending']);

function injectStyles(): void {
    const style = document.createElement('style');
    style.textContent = css;
    document.head.appendChild(style);
}

function mount(): Root {
    injectStyles();
    let el = document.getElementById('root');
    if (!el) {
        el = document.createElement('div');
        el.id = 'root';
        document.body.appendChild(el);
    }
    return createRoot(el);
}

/** Placeholder shown while a tensor is on its way, or being read. */
function Waiting({ name, what }: { name?: string; what: string }) {
    return (
        <div className="ptv-root">
            <div className="ptv-header">
                <h2 className="ptv-title">{name ?? 'tensor'}</h2>
            </div>
            <div className="ptv-meta">{what}</div>
        </div>
    );
}

/** One decoder for the life of the panel: it owns the worker. */
const decoder = createDecoder({ warn: (message) => console.warn(message) });

/**
 * Reads and decodes the tensor at `url`, then renders it. Remounted per URL by
 * its key, so a tensor that arrives while an older one is still decoding
 * replaces it instead of racing it.
 */
function TensorFromUrl({ message, onRefresh }: { message: TensorMessage; onRefresh?: () => void }) {
    const [tensor, setTensor] = useState<Tensor | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        let cancelled = false;
        decoder.decode(message.url).then(
            (decoded) => {
                if (!cancelled) {
                    setTensor(decoded);
                }
            },
            (failure: unknown) => {
                if (!cancelled) {
                    setError(failure instanceof Error ? failure.message : String(failure));
                }
            }
        );
        return () => {
            cancelled = true;
        };
    }, [message.url]);

    if (error !== null) {
        return <Waiting name={message.name} what={`could not read the tensor: ${error}`} />;
    }
    if (tensor === null) {
        return <Waiting name={message.name} what="reading the tensor…" />;
    }
    return (
        <TensorViewer
            tensor={tensor}
            name={message.name}
            tableThreshold={message.tableThreshold}
            onRefresh={onRefresh}
        />
    );
}

const root = mount();
const vscode = window.acquireVsCodeApi?.();

function render(msg: HostMessage): void {
    const body =
        msg.type === 'pending' ? (
            <Waiting name={msg.name} what="waiting for the debuggee to send it…" />
        ) : msg.type === 'demo' ? (
            <SampleBrowser samples={SAMPLES} tableThreshold={msg.tableThreshold} />
        ) : (
            <TensorFromUrl
                key={msg.url}
                message={msg}
                onRefresh={
                    msg.canRefresh && vscode
                        ? () => vscode.postMessage({ type: 'refresh' })
                        : undefined
                }
            />
        );
    root.render(<StrictMode>{body}</StrictMode>);
}

window.addEventListener('message', (event: MessageEvent) => {
    const msg = event.data as HostMessage | undefined;
    if (msg && KINDS.has(msg.type)) {
        render(msg);
    }
});

// First paint from the embedded init message (no round-trip).
const initial = window.__PTENSOR_INIT__;
if (initial && KINDS.has(initial.type)) {
    render(initial);
}

// Tell the host we are ready (used for subsequent updates when the panel is reused).
vscode?.postMessage({ type: 'ready' });
