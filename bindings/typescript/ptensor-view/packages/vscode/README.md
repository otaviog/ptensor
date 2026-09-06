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

- Run `npm install` in `bindings/typescript/ptensor-vscode` once beforehand (links the `file:`
  dependencies on `ptensor-ts` and `ptensor-tlog`, and pulls in `@vscode/vsce`).
- The `code` CLI must be on `PATH` (VS Code → *Shell Command: Install 'code'
  command*).
- Requires VS Code ≥ 1.120 (see `engines.vscode`).
- `cmake --install` also copies `build/ptensor-vscode.vsix` into
  `<prefix>/share/ptensor` when the file exists (build it first).

### B. Run from source (dev, hot iterate)

Open `bindings/typescript/ptensor-vscode` in VS Code and press **F5** ("Run Extension") to launch
an Extension Development Host with the extension loaded. Run
`npm run build/assets` once first so the webview bundle and the icon exist in
`media/`.

## Try it without a debugger

Run **`ptensor: Preview Sample Tensors`** from the command palette. It opens the
viewer panel with built-in sample tensors (tables, grayscale / RGB images,
planar vs interleaved, batched) and a sidebar to switch between them — the
fastest way to see the panel. For pure UI work, `bun run dev` in
`bindings/typescript/ptensor-view` serves the same components with HMR.

## What it does

While stopped at a breakpoint, right-click any `Tensor` variable in the
**Variables** view and pick **Visualize Tensor**, or run
`ptensor: Visualize Tensor` from the command palette and type the expression.

The extension listens on an ephemeral loopback port and asks the debuggee to
send the tensor there:

```cpp
p10::tlog::log_to("127.0.0.1:<port>", "<expr>", <expr>)
```

That is the whole debugger interaction. The tensor travels as bytes over TCP --
the same `p10::tlog` wire format the desktop viewer consumes -- so its size no
longer depends on what the debugger's expression printer can render. (It used to
be read back as a JSON string from `p10::to_json_debug`, which LLDB truncates at
`target.max-string-summary-length` and reads back a chunk at a time; a large
tensor was slow at best.)

The tab opens immediately and fills in when the tensor lands, showing:

- **min / max / mean / count** stats over all elements.
- **Table view** for small tensors (element count <= `ptensor.tableElementThreshold`, default 256).
- **Image view** when the shape looks image-like:
  - `[H, W]` -- grayscale
  - `[H, W, C]` interleaved with `C in {1, 3, 4}`
  - `[C, H, W]` planar with `C in {1, 3, 4}`
  - `[N, C, H, W]` or `[N, H, W, C]` -- one tab per `N`
- Float tensors are window-stretched to `[min, max]` for display; `uint8` is shown as-is.

**Refresh** re-runs the same call against the current frame, so stepping and
hitting refresh shows the tensor as it is now.

## Requirements

- A C++ debugger that supports DAP `evaluate` with function calls (lldb-dap,
  CodeLLDB, cppdbg).
- The debuggee must be paused at a frame where the tensor expression resolves.
- **The debuggee must link `ptensor_tlog` and keep the symbol.** The debugger
  can only call `p10::tlog::log_to` if it is in the binary, and a program that
  never logs will have it stripped. Reference it once, as
  `src/tlog/tests/utils/viewer_demo.cpp` does:

  ```cpp
  // Anchors the symbol; nothing listens on port 1, and tlog swallows that.
  p10::tlog::log_to("127.0.0.1:1", "linker-anchor", some_tensor);
  ```

  Without it the command reports that `log_to` is not linked into the debuggee.

`log_to` takes `const char*` rather than `std::string` precisely so the
expression evaluator can pass the literals: LLDB evaluates as C++14 and will not
construct a `std::string` for a `const std::string&` parameter.

## Settings

- `ptensor.tableElementThreshold` (default `256`) -- element count above which the panel prefers an image view.

## Known limitations

- No live updates: the panel reflects the tensor at the moment the command was
  run (or last refreshed).
- `Float16` is converted to `Float32` for display.
- Strides are ignored -- assumes contiguous data.
- The image colormap stretches floats by global min/max; no per-channel mapping yet.
- The feed accepts any connection on its loopback port while a debug session is
  open, and shows what arrives. It is a local debugging aid, not an authenticated
  channel.
