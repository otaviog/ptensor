// Bundles the extension (and its lone runtime dependency, ptensor-ts) into a
// single self-contained out/extension.js. This lets the .vsix ship without
// node_modules and lets `vsce package --no-dependencies` skip the npm dependency
// walk, which trips over the file:-linked ptensor-ts's dev-only deps.
import { rm } from 'node:fs/promises';
import { build, context } from 'esbuild';

const watch = process.argv.includes('--watch');

// Drop any stale tsc output so the .vsix ships only the bundle.
await rm('out', { recursive: true, force: true });

const options = {
    entryPoints: ['src/extension.ts'],
    bundle: true,
    outfile: 'out/extension.js',
    platform: 'node',
    format: 'cjs',
    target: 'node18',
    // Provided by the VS Code runtime, never bundled.
    external: ['vscode'],
    sourcemap: !process.argv.includes('--minify'),
    minify: process.argv.includes('--minify'),
    logLevel: 'info',
};

if (watch) {
    const ctx = await context(options);
    await ctx.watch();
    console.log('esbuild watching…');
} else {
    await build(options);
    console.log('extension bundled -> out/extension.js');
}
