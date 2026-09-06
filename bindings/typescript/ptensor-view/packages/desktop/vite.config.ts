// Dev server for the window (`bun run dev:ui`). The shipped app still loads the
// bundle electrobun builds; this only serves src/webview with React Fast
// Refresh so UI edits land in the running window without restarting the bun
// process -- the socket feed, and the tensors already received, survive.
//
// Point the app at it with PTENSOR_VIEW_DEV_URL (see src/bun/index.ts), or just
// run `bun run dev:hot`.

import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const packageRoot = fileURLToPath(new URL('.', import.meta.url));

export default defineConfig({
    root: 'src/webview/dev',
    plugins: [react()],
    resolve: {
        alias: {
            // The published entry points at ptensor-view's build output, which
            // would freeze the viewer at whatever was last built. Its sources
            // hot-reload like the app's own.
            '@ptensor/tensor-view/styles.css': `${packageRoot}../view/src/styles.css`,
            '@ptensor/tensor-view': `${packageRoot}../view/src/index.ts`,
        },
    },
    server: {
        port: 5174,
        strictPort: true,
        fs: {
            // The workspace packages live outside the Vite root.
            allow: [`${packageRoot}..`],
        },
    },
});
