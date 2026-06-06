// Webview bootstrap. Built by Vite (lib/iife) into dist/webview.js and loaded
// by the VS Code extension. Mounts the React panel and waits for the host to
// post a tensor as `TensorJson` (the debugger's own JSON shape).

import { StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { SampleBrowser } from '../components/SampleBrowser';
import { TensorViewer } from '../components/TensorViewer';
import { SAMPLES } from '../samples';
import { type TensorJson } from 'ptensor-ts';
import { fromTensorJson } from '../tensorView';
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

/** Message posted by the extension host. `tensor` is the debugger JSON. */
interface TensorMessage {
    type: 'tensor';
    name?: string;
    tensor: TensorJson;
    tableThreshold?: number;
    /** Whether the host can re-read this tensor (false for demo/sample tensors). */
    canRefresh?: boolean;
}

/** Demo mode: render the built-in sample tensors, no debugger needed. */
interface DemoMessage {
    type: 'demo';
    tableThreshold?: number;
}

type HostMessage = TensorMessage | DemoMessage;

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

const root = mount();
const vscode = window.acquireVsCodeApi?.();

function render(msg: HostMessage): void {
    const body =
        msg.type === 'demo' ? (
            <SampleBrowser samples={SAMPLES} tableThreshold={msg.tableThreshold} />
        ) : (
            <TensorViewer
                tensor={fromTensorJson(msg.tensor, msg.name)}
                tableThreshold={msg.tableThreshold}
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
    if (msg?.type === 'tensor' || msg?.type === 'demo') {
        render(msg);
    }
});

// First paint from the embedded init message (no round-trip).
const initial = window.__PTENSOR_INIT__;
if (initial && (initial.type === 'tensor' || initial.type === 'demo')) {
    render(initial);
}

// Tell the host we are ready (used for subsequent updates when the panel is reused).
vscode?.postMessage({ type: 'ready' });
