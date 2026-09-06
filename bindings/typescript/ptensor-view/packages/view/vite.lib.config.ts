import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import dts from 'vite-plugin-dts';

// Library build: the reusable component/type API (src/index.ts) for the
// Electron pilot and other Node/bundler consumers. React stays external here
// (consumers provide it); `emptyOutDir: false` keeps the webview build output.
export default defineConfig({
    plugins: [react(), dts({ include: ['src'], rollupTypes: true })],
    build: {
        outDir: 'dist',
        emptyOutDir: false,
        lib: {
            entry: 'src/index.ts',
            formats: ['es'],
            fileName: () => 'tensor-view.js',
        },
        rollupOptions: {
            external: ['react', 'react-dom', 'react/jsx-runtime'],
        },
    },
});
