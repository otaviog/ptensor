# ptensor viewer workspace

One install, one lockfile, four packages:

| package | what it is |
| --- | --- |
| `packages/view` (`@ptensor/tensor-view`) | the React panel: tables, images, batched image tabs |
| `packages/tlog` (`ptensor-tlog`) | the `p10::tlog` wire protocol and the sockets that serve it: `ptensor-tlog` is framing and parsing with no runtime, `ptensor-tlog/bun` and `ptensor-tlog/node` are the two servers |
| `packages/desktop` (`ptensor-desktop`) | the Electrobun app: a window fed by a local socket |
| `packages/vscode` (`ptensor-vscode`) | the extension: a viewer tab fed by the debuggee |

`ptensor-ts` and `ptensor-ffi` stay their own projects, one directory up. That
is why `bun run build` starts by building `ptensor-ts` and reinstalling: a
`file:` dependency outside the workspace is snapshotted at install time, so its
`dist` would otherwise be whatever it was when you last installed. The four
packages here are symlinked to each other, so they never have that problem.

```bash
bun install          # once, at this root
bun run build        # ptensor-ts, then view and tlog (the apps consume their dist)
bun run typecheck
bun run test
```

Package order matters for the first build: `desktop` and `vscode` typecheck
against `view`'s and `tlog`'s build output, so `bun run build` before either.
Within the workspace the packages are symlinked, so a rebuild is picked up
without reinstalling.
