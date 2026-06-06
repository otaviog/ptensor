# ptensor Tensor Viewer

VS Code debugger helper that visualizes `p10::Tensor` contents while a C++ debug session is paused.

## Install

### Build a `.vsix`

From the repo root:

```bash
just vscode-package
code --install-extension build/ptensor-vscode.vsix
```

`just vscode-package` writes `build/ptensor-vscode.vsix` (vsce's prepublish step
builds the viewer bundle and compiles the extension first). Reload VS Code after
installing.

Notes:

- Run `npm install` in `src/ptensor-vscode` once beforehand (links the `file:`
  dependency on `ptensor-ts` and pulls in `@vscode/vsce`).
- The `code` CLI must be on `PATH` (VS Code → *Shell Command: Install 'code'
  command*).
- Requires VS Code ≥ 1.120 (see `engines.vscode`).
- `cmake --install` also copies `build/ptensor-vscode.vsix` into
  `<prefix>/share/ptensor` when the file exists (build it first).

### B. Run from source (dev, hot iterate)

Open `src/ptensor-vscode` in VS Code and press **F5** ("Run Extension") to launch
an Extension Development Host with the extension loaded. Run
`just view-build` once first so the webview bundle exists in `media/`.

## Try it without a debugger

Run **`ptensor: Preview Sample Tensors`** from the command palette. It opens the
viewer panel with built-in sample tensors (tables, grayscale / RGB images,
planar vs interleaved, batched) and a sidebar to switch between them — the
fastest way to see the panel. For pure UI work, `bun run dev` in
`src/ptensor-view` serves the same components with HMR.

## What it does

While stopped at a breakpoint, right-click any `Tensor` variable in the **Variables** view and pick **Visualize Tensor**, or run `ptensor: Visualize Tensor` from the command palette and type the expression. The extension queries the debug session for the tensor's shape, dtype, and data pointer, reads the raw bytes via DAP `readMemory`, and opens a webview that shows:

- **min / max / mean / count** stats over all elements.
- **Table view** for small tensors (element count ≤ `ptensor.tableElementThreshold`, default 256).
- **Image view** when the shape looks image-like:
  - `[H, W]` — grayscale
  - `[H, W, C]` interleaved with `C ∈ {1, 3, 4}`
  - `[C, H, W]` planar with `C ∈ {1, 3, 4}`
  - `[N, C, H, W]` or `[N, H, W, C]` — one tab per `N`
- Float tensors are window-stretched to `[min, max]` for display; `uint8` is shown as-is.

## Requirements

- A C++ debugger that supports DAP `evaluate` and `readMemory`. Tested mentally with `cppdbg`, `cppvsdbg`, and CodeLLDB.
- The debuggee must be paused at a frame where the tensor expression resolves.

The extension evaluates these expressions against the focused frame:

```cpp
(int)((expr).dims())
(long long)((expr).shape().as_span().data()[i])
(int)((expr).dtype().value)
(unsigned long long)((expr).size_bytes())
(expr).as_bytes().data()    // for memoryReference
```

If your build inlines these away, set the optimization level low enough that the debugger can call them.

## Settings

- `ptensor.tableElementThreshold` (default `256`) — element count above which the panel prefers an image view.
- `ptensor.maxBytes` (default `64 MiB`) — hard cap on bytes read from the debuggee.

## Known limitations

- No live updates: the panel reflects the tensor at the moment the command was run.
- `Float16` is converted to `Float32` for display.
- Strides are ignored — assumes contiguous data (matches `Tensor::as_bytes()`).
- The image colormap stretches floats by global min/max; no per-channel mapping yet.
