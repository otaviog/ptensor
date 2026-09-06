import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// Default config drives two things:
//  - `vite dev`   -> serves index.html (the dev playground) with HMR.
//  - `vite build` -> bundles the webview entry into a single self-contained
//                    IIFE (React inlined) that the VS Code extension loads.
export default defineConfig({
    plugins: [react()],
    // Lib mode leaves `process.env.NODE_ENV` unreplaced (it expects the
    // consumer's bundler to handle it). A VS Code webview has no `process`
    // global, so React would throw `process is not defined` on load and the
    // panel would render blank. Replace it at build time and pick React's
    // production path.
    define: {
        'process.env.NODE_ENV': JSON.stringify('production'),
    },
    build: {
        outDir: 'dist',
        emptyOutDir: true,
        cssCodeSplit: true,
        lib: {
            entry: 'src/webview/main.tsx',
            formats: ['iife'],
            name: 'PtensorTensorView',
            fileName: () => 'webview.js',
        },
        rollupOptions: {
            output: {
                assetFileNames: 'webview.[ext]',
            },
        },
    },
});
