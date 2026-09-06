# @ptensor/tensor-view

Framework-agnostic tensor visualization for ptensor: a React panel that renders
a tensor as a table, an image (grayscale / RGB, planar or interleaved), or
batched image tabs.

Pure: no native / FFI dependency, so the same code runs in a VS Code webview,
the Electron pilot, and a plain browser playground.

## The tensor

This package owns no tensor vocabulary at all: the type, its dtypes and the
helpers for reading one (`elementAt`, `isFloatDtype`, `dtypeSizeBytes`) are
ptensor-ts', re-exported from this entry so the panel and what it takes are one
import. The panel renders the same decoded, plain-data tensor the rest of the TS
side passes around:

```ts
interface Tensor {
    dtype: DTypeString;
    shape: number[];
    stride: number[];   // element counts
    data: NumericArray; // int64 -> BigInt64Array
}
```

Tensors cross process boundaries as `TensorJson` (`{dtype, shape, stride, blob}`,
the exact shape `p10::to_json_debug` emits, with `blob` base64 or base64+zstd).
Decoding it is ptensor-ts' job -- `tensorFromJson`, re-exported here so the
panel and its transport are one import. A float16 payload comes back as a
`Float32Array`, since readers want numbers rather than raw bits, while `dtype`
keeps saying `float16`.

The header label is a prop, not a field: `<TensorViewer tensor={t} name="input" />`.

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

`dist/tensor-view.js` is what `../ptensor-desktop` (the Electrobun app that
streams tensors in over a local socket) imports; its stylesheet is exported
separately as `@ptensor/tensor-view/styles.css`, since the lib build does not
bundle CSS.
