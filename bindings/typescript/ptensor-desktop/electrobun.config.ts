import type { ElectrobunConfig } from 'electrobun';
import { dedupeReact } from './src/build/dedupeReact';

export default {
    app: {
        name: 'ptensor View',
        identifier: 'com.ptensor.tensor-view',
        version: '0.1.0',
        description: 'Views tensors streamed to a local socket, grouped by session id.',
    },
    build: {
        bun: {
            entrypoint: 'src/bun/index.ts',
        },
        views: {
            mainview: {
                entrypoint: 'src/mainview/index.tsx',
                plugins: [dedupeReact],
            },
        },
        copy: {
            'src/mainview/index.html': 'views/mainview/index.html',
        },
        mac: {
            // System WebKit: no CEF download, and the panel needs nothing extra.
            bundleCEF: false,
            codesign: false,
            notarize: false,
            createDmg: false,
        },
    },
} satisfies ElectrobunConfig;
