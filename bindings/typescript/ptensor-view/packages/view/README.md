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

## Decoding, off the calling thread

`@ptensor/tensor-view/decode` is the other half of this package, and pulls no
React: give it a URL that answers with a `TensorJson` and it hands back a
`Tensor`.

```ts
import { createDecoder } from '@ptensor/tensor-view/decode';

const decoder = createDecoder({ warn: (m) => console.warn(m) });
const tensor = await decoder.decode('http://127.0.0.1:5123/tensor/frame?token=…');
```

It exists because base64 plus zstd on a batched float32 image
(`30x3x570x1132`) is about 1.3 s of straight-line JS. On a UI thread that is
1.3 s of frozen panel per tensor, so it runs in a worker, which does the
*fetch* as well as the decode -- the tensor's 262 MiB of base64 never enters
the caller's heap -- and transfers the decoded buffer back rather than copying
it.

If the worker cannot start (no `Worker`, blob URLs refused, a bundle the engine
will not parse) decoding falls back to the calling thread: slower, but working.
`TensorDecoder.usesWorker` says which is in use, and the tests assert it,
because otherwise the fallback passes every test and the only sign is a line in
the log.

### The worker is a generated source string

The worker cannot be a bundler entrypoint: `Bun.build` leaves
`new Worker(new URL('./x.ts', import.meta.url))` exactly as written and emits no
chunk for it, so at runtime the host asks for a `.ts` file. So
`scripts/buildDecodeWorker.ts` bundles `src/decode/decodeWorker.ts` into a
module exporting its source as a string (9.6 kB), which `createDecoder` starts
from a blob URL. A string also avoids the import attribute that Vite and
electrobun's `Bun.build` spell differently.

The generated file is git-ignored and `build`, `typecheck` and `test` rebuild it
first. Hosts need no build step of their own -- the string is inlined into
`dist/decode.js`.

One trap worth knowing: the blob URL is revoked on `dispose`, not right after
`new Worker(...)`. A worker fetches its script asynchronously, so revoking
immediately races it -- bun refuses outright with "Blob URL is missing", and a
browser is not obliged to do better.

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
bun run build:lib       # dist/tensor-view.js + dist/decode.js + their .d.ts
bun run build           # both
bun run worker          # just the decode worker's source string
```

`dist/webview.js` is loaded by the `ptensor-vscode` extension: it mounts the
panel, posts a `ready` message, and renders the tensor the host posts back.

`dist/tensor-view.js` is what `../desktop` (the Electrobun app that streams
tensors in over a local socket) imports; its stylesheet is exported separately
as `@ptensor/tensor-view/styles.css`, since the lib build does not bundle CSS.

`dist/decode.js` is the second entry, `@ptensor/tensor-view/decode`. It is
separate because it is not the same dependency: importing the decoder should not
drag React in, and a host that only moves tensors around never needs the panel.
