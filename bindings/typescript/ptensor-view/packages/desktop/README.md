# @ptensor/tensor-desktop

An [Electrobun](https://electrobun.dev) desktop app that shows tensors streamed
to a local socket. Every tensor carries a session id, and the window groups the
incoming tensors by session: pick a session on the left, pick a tensor, and the
`ptensor-view` panel renders it (table / grayscale / RGB / batched images).

```
producer ──TCP 127.0.0.1:8791, newline-delimited JSON──▶ bun process ──metadata push (ws)──▶ webview
   (C++ / Python / TS)                                   socket + framing       ◀──GET /tensor/<id>──  TensorViewer
                                                         + the tensor store
```

The bun process holds each tensor as the `TensorJson` text it arrived as --
still base64, still zstd-compressed -- and the window reads the one it is
showing over HTTP. What crosses as a message is only metadata: a few hundred
bytes saying a tensor exists.

That split is not an optimisation, it is the difference between working and
not. A batched float32 image (`30x3x570x1132`) is 232 MB of elements and some
150 MB of base64, and every message channel between a host process and a
webview stringifies. Electrobun's encrypted RPC decodes its base64 with
`atob(s).split('').map(...)` in the webview preload, which costs ~17 bytes of
heap per decoded byte: two of those tensors pushed as messages is 12 GB of
webview memory and a frozen window. Over `fetch` the same base64 is one
allocation and about 100 ms.

## Run it

```bash
bun install
bun run build          # builds build/<channel>/ptensor View-dev.app
bun run start          # runs that app

# in another shell: feed it some tensors
bun run send -- --session demo --count 20 --interval 200 --kind rgb
```

`bun run dev` does the same through Electrobun's dev loop: it bundles the app
and launches it, and every change needs another run.

### Working on the UI

```bash
bun run dev/ui         # vite on :5174, plus the app pointed at it
```

The window loads from the Vite dev server instead of the bundle, so edits to the
webview -- this package's `src/webview`, and ptensor-view's sources, which are
aliased to source rather than its `dist` -- land in the open window through
React Fast Refresh. The bun process is not restarted, so the socket keeps
listening and the tensors already in the window stay there.

The tensors live in the bun process, which is not restarted, so a remount --
an edit that adds or removes a hook -- no longer loses them: the panel
re-reads the listing when its socket reconnects.

It is the dev server that decides this, not the script: the app loads whatever
`PTENSOR_VIEW_DEV_URL` names, and the bundled view -- the one that ships -- when
it is unset. Set it yourself to point the app at a server on another port.
`scripts/devUi.ts` only waits for Vite before starting the app (the window loads
its URL once and does not retry) and stops it afterwards (the port is
`strictPort`, so a stray server breaks the next run).

Only the window is served this way -- a change to `src/bun` still needs a
restart, and `bunx electrobun dev --watch` covers that by rebuilding and
relaunching the whole app on every change.

### App icon

There is one icon in the repo, `docs/icon-light.png`. This package keeps no copy
of it: `scripts/makeIcons.ts` cuts every size the three platforms need into
`.icons/`, which is git-ignored, and `bun run dev` / `bun run build` /
`bun run dev/ui` run it first. It skips the work when the output is newer than
the source; `bun run icons -- --force` re-cuts anyway.

| platform | file | how it is used |
| --- | --- | --- |
| macOS | `.icons/icon.iconset/` (10 PNGs, 16-1024) | `iconutil` compiles it into the bundle's `AppIcon.icns` |
| Windows | `.icons/icon.ico` (16/24/32/48/64/128/256) | embedded in `launcher.exe`, copied to `Resources/app.ico` |
| Linux | `.icons/icon.png` (512) | copied to `Resources/appIcon.png` |

The cutting is pure JS (`pngjs`, plus png-to-ico's bicubic resize), not `sips`
and `iconutil`, so the Windows and Linux icons can be cut on those platforms;
only the macOS `.iconset` -> `.icns` step needs macOS. To use `icon-dark.png`
instead, point `SOURCE` in the script at it.

The source is 250x251 -- squared onto a transparent canvas before resizing --
so anything above 256 is upscaled. Replace it if a larger original turns up.

Only the macOS icon is verified here: the Windows and Linux ones are wired in
the config but need a build on those platforms to see.

The first build downloads Electrobun's platform binaries (~28 MB) into
`node_modules/electrobun`.

## Wire protocol

Connect to `127.0.0.1:8791` and write one JSON object per line (`\n`
terminated). Two kinds of line are accepted, told apart by their fields.

A session line names the session everything after it lands in. Send it once,
first:

```json
{"sessionId": "eulerian"}
```

The id is fixed for the life of the connection: a second session line is
logged and ignored, and a connection that sends none files its tensors under
`default`. Use another connection for another session.

A tensor line carries one tensor:

```json
{"name": "frame 12", "tensor": {"dtype": "uint8", "shape": [96,128,3], "stride": [384,3,1], "blob": "<base64>"}}
```

- `name` — label in the tensor list.
- `tensor` — exactly what `p10::to_json_debug` emits: `{dtype, shape, stride,
  blob}` with `blob` the raw little-endian element bytes, base64-encoded. So a
  C++ producer can write `to_json_debug(tensor)` straight into the line.

Many tensors per connection are fine, and a connection can stay open for the
life of a run. The ✕ on a session, or `Clear all sessions`, asks the bun
process to drop them.

Lines that are not valid JSON, or that fail validation, are logged and skipped —
the connection stays up. A line over `PTENSOR_VIEW_MAX_LINE_MB` (1 GiB) closes
it.

That default is not arbitrary. A `30x3x570x1132` float32 tensor is 222 MiB of
elements, and float mantissas barely compress — 1.13x at zstd -1 — so its line
is about 262 MiB, past the 256 MiB `ptensor-tlog` defaults to. At that cap the
feed drops the connection instead of showing the tensor.

A minimal Python producer:

```python
import base64, json, socket
import numpy as np

img = (np.random.rand(96, 128, 3) * 255).astype(np.uint8)
lines = [
    {"sessionId": "py-demo"},
    {
        "name": "noise",
        "tensor": {
            "dtype": "uint8",
            "shape": list(img.shape),
            "stride": [s // img.itemsize for s in img.strides],
            "blob": base64.b64encode(img.tobytes()).decode(),
        },
    },
]
with socket.create_connection(("127.0.0.1", 8791)) as sock:
    for line in lines:
        sock.sendall((json.dumps(line) + "\n").encode())
```

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `PTENSOR_VIEW_PORT` | `8791` | Port the feed listens on |
| `PTENSOR_VIEW_HOST` | `127.0.0.1` | Interface the feed binds to |
| `PTENSOR_VIEW_BUDGET_MB` | `2048` | Tensor JSON the bun process holds, in MiB (oldest dropped) |
| `PTENSOR_VIEW_MAX_LINE_MB` | `1024` | Largest single wire line accepted, in MiB |
| `PTENSOR_VIEW_LOG_LEVEL` | `info` | `debug`, `info`, `warning`, `error` or `fatal` |
| `PTENSOR_VIEW_LOG_FILE` | platform default | Explicit log file path |

If the port is taken the app still opens and the sidebar shows `feed down: …`.

## Logs

The main process logs through [LogTape](https://logtape.org) to the console and
to a rotating file (5 MiB, 5 generations kept):

| Platform | Log file | Fallback when that directory is not writable |
| --- | --- | --- |
| Windows | `%PROGRAMDATA%\ptensor\ptensor-desktop.log` | `%LOCALAPPDATA%\ptensor\logs\` |
| macOS | `/Library/Logs/ptensor/ptensor-desktop.log` | `~/Library/Logs/ptensor/` |
| Linux | `/var/log/ptensor/ptensor-desktop.log` | `~/.local/state/ptensor/` |

If neither location can be opened the app still starts and logs to the console
only. Webview logs stay in the webview console; only the main process writes to
the file.

## Layout

The directories are named after electrobun's two sides, `bun` and `webview`.

- `src/bun/` — main process: `index.ts` (store, window, wiring),
  `feedServer.ts` (the HTTP + WebSocket server the window reads) and
  `logFile.ts` (rotating log file).
- `src/webview/` — the window: `App.tsx` (grouping, selection, follow mode),
  `feedClient.ts` (socket + fetch + decode), `tensorCache.ts` (byte-capped
  cache of decoded tensors), `index.tsx` (style injection).
- `src/shared/` — `feed.ts` (the bun ⇄ webview contract) and `logging.ts`
  (LogTape setup).
- `scripts/sendTensor.ts` — synthetic producer used for manual testing.

The feed and the framing live in `ptensor-tlog`, which also owns the store
(`TensorStore`): the tensors, keyed by a URL-safe id derived from the
producer's label, under a byte budget rather than a tensor count -- one batched
image is a couple of hundred megabytes, so "keep the last 100" is not a thing
that can be done.

### Reading the feed

Three routes, all on loopback with an OS-picked port, all carrying a `token`
minted at startup:

| Route | Answers |
| --- | --- |
| `GET /tensors` | every held tensor's metadata, newest first |
| `GET /tensor/<id>` | that tensor's `TensorJson`, byte for byte as it arrived |
| `ws /feed` | a `hello` with the feed state and the listing, then one event per arrival |

The window finds the address on `window.__ptensorFeed`, injected by a preload
script. Under `bun run dev/ui` the window URL's query carries it too, as a
fallback; the bundled app cannot use that channel, because the `views://`
scheme handler resolves a URL to a bundled file and `index.html?feed=...`
matches none, which loads an empty window.

Because it is plain HTTP, the feed is testable without electrobun and without a
window (`src/bun/__tests__/feedServer.test.ts`), and the panel served by
`bun run dev/ui` reads real tensors.

The window decodes only the tensor it is showing and keeps the decoded ones in
a 1 GiB LRU, so clicking back and forth through a session costs nothing and
cannot grow without bound.

### The decode worker

base64 plus zstd on a batched float32 image is about 1.3 s of straight-line JS,
which on the window's thread is 1.3 s of frozen panel per tensor. So it runs in
a worker (`src/webview/decoder.ts`), which does the *fetch* as well as the
decode -- the tensor's 262 MiB of base64 never enters the window's heap -- and
transfers the decoded buffer back rather than copying it.

The worker cannot be a second bundler entrypoint: `Bun.build` leaves
`new Worker(new URL('./x.ts', import.meta.url))` exactly as written and emits no
chunk for it. So `scripts/buildDecodeWorker.ts` bundles it into a module that
exports its source as a string (9.6 kB), and the window starts it from a blob
URL. A source string also avoids the import attribute that electrobun's
`Bun.build` and Vite spell differently. The generated file is git-ignored; the
`dev`, `dev/ui`, `build`, `typecheck` and `test` scripts all rebuild it first.

If the worker cannot start, decoding falls back to the window's thread --
slower, but working. `TensorDecoder.usesWorker` says which is in use, and the
decoder tests assert it, because otherwise the fallback passes every test and
the only sign is a line in the log.

## Checks

```bash
bun run typecheck   # tsc over the app sources
bun test            # feed server, feed client, App behaviour, tensor cache
```

The feed server and client tests talk to a real socket, which took two things
to make work in the same process as the panel tests: the DOM preload puts back
the real `fetch` and `WebSocket` after happy-dom installs its emulations, and
the panel's `mock.module` of `@ptensor/tensor-view` -- process-wide, like every
`mock.module` -- carries the real `tensorFromJson` so the client test decodes a
real blob.

## Note on React

`@ptensor/tensor-view` is linked with `file:` and keeps its own `node_modules`,
so React would end up in the bundle twice (and every hook in `TensorViewer`
would throw). `src/build/dedupeReact.ts` is a Bun plugin that pins every
`react` / `react-dom` specifier to this app's copy; it is wired into the view
build in `electrobun.config.ts`. Turning `bindings/typescript` into a Bun
workspace would remove the need for it.
