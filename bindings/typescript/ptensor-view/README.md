# @ptensor/tensor-view

Framework-agnostic tensor visualization for ptensor. Owns the shared
`TensorView` type and a React panel that renders a tensor as a table, an image
(grayscale / RGB, planar or interleaved), or batched image tabs.

Pure: no native / FFI dependency, so the same code runs in a VS Code webview,
the Electron pilot, and a plain browser playground.

## TensorView

```ts
interface TensorView {
    array: NumericArray;   // decoded; float16 -> Float32Array, int64 -> BigInt64Array
    stride: bigint[];
    shape: bigint[];
    dtype: DTypeString;
    name?: string;
}
```

Tensors cross process boundaries as `TensorJson` (`{dtype, shape, stride, blob}`,
the exact shape `p10::to_json_debug` emits, with `blob` base64); `fromTensorJson`
decodes it — including float16 — into a `TensorView`.

## Develop

```bash
bun install
bun run dev          # Vite playground with mock tensors + HMR (src/ptensor-view/dev)
bun run typecheck
```

The playground (`dev/`) renders every `resolveView` branch from `dev/samples.ts`,
which mirror the C++ `vscode_viewer_demo` driver and the live debugger path.

## Build

```bash
bun run build:webview   # dist/webview.js — single self-contained IIFE (React + CSS inlined)
bun run build:lib       # dist/tensor-view.js + index.d.ts — reusable component/type API
bun run build           # both
```

`dist/webview.js` is loaded by the `ptensor-vscode` extension: it mounts the
panel, posts a `ready` message, and renders the tensor the host posts back.
