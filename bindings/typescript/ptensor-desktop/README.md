# @ptensor/tensor-desktop

An [Electrobun](https://electrobun.dev) desktop app that shows tensors streamed
to a local socket. Every tensor carries a session id, and the window groups the
incoming tensors by session: pick a session on the left, pick a tensor, and the
`ptensor-view` panel renders it (table / grayscale / RGB / batched images).

```
producer ──TCP 127.0.0.1:8791, newline-delimited JSON──▶ bun process ──RPC push──▶ webview
   (C++ / Python / TS)                                   socket + framing        history + TensorViewer
```

The bun process stores nothing: it validates each line and pushes the tensor to
the window, which keeps the history it shows.

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

The window keeps its tensor history in React state, so an edit that adds or
removes a hook remounts the component and clears the list -- send the tensors
again. Edits that leave the hooks alone keep it.

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
life of a run. History is dropped from the window itself (the ✕ on a session,
or `Clear all sessions`).

Lines that are not valid JSON, or that fail validation, are logged and skipped —
the connection stays up. A single line over 256 MiB closes the connection.

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
| `PTENSOR_VIEW_HISTORY` | `100` | Tensors kept per session (oldest dropped) |
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

The directories are named after electrobun's two sides, `bun` and `webview`,
which are also the keys of the RPC schema.

- `src/bun/` — main process: `index.ts` (window, RPC handlers, pushes) and
  `logFile.ts` (rotating log file).
- `src/bun/server/` — the feed: `tensorServer.ts` (socket), `lineSplitter.ts`
  (framing), `protocol.ts` (wire types + validation, shared with producers),
  `connectionHandler.ts` (per-connection session state).
- `src/webview/` — the window: `App.tsx` (history, grouping, selection, follow
  mode), `index.tsx` (RPC wiring, style injection).
- `src/shared/` — `rpc.ts` (the bun ⇄ webview contract) and `logging.ts`
  (LogTape setup).
- `scripts/send-tensor.ts` — synthetic producer used for manual testing.

The window keeps the history and decodes only the selected tensor, so a fast
producer costs base64 text, not decoded buffers. `PTENSOR_VIEW_HISTORY` is read
by the main process and travels to the window in the `getServerInfo` answer;
until that answer lands the window uses 100, then trims to what was reported.

## Checks

```bash
bun run typecheck   # tsc over the app sources
bun test            # framing, protocol, connection handling, App behaviour
bun run smoke       # builds the real view bundle and drives it headlessly
```

`bun run smoke` is the one that catches bundling problems: it renders the
shipped bundle in a DOM, answers its RPC with a fake host, and asserts the
panel follows pushes.

## Note on React

`@ptensor/tensor-view` is linked with `file:` and keeps its own `node_modules`,
so React would end up in the bundle twice (and every hook in `TensorViewer`
would throw). `src/build/dedupeReact.ts` is a Bun plugin that pins every
`react` / `react-dom` specifier to this app's copy; it is wired into the view
build in `electrobun.config.ts`. Turning `bindings/typescript` into a Bun
workspace would remove the need for it.
