import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import dts from 'vite-plugin-dts';

// Library build: the reusable API for the apps that host the panel. Two
// entries, because they are not the same dependency: `tensor-view` is the React
// panel, `decode` is the tensor decoder and pulls no React, so an extension
// host or a worker can import it without the renderer.
//
// React stays external here (consumers provide it); `emptyOutDir: false` keeps
// the webview build output.
export default defineConfig({
    plugins: [react(), dts({ include: ['src'], rollupTypes: true })],
    build: {
        outDir: 'dist',
        emptyOutDir: false,
        lib: {
            entry: {
                'tensor-view': 'src/index.ts',
                decode: 'src/decode/index.ts',
            },
            formats: ['es'],
            fileName: (_format, name) => `${name}.js`,
        },
        rollupOptions: {
            external: ['react', 'react-dom', 'react/jsx-runtime'],
        },
    },
});
