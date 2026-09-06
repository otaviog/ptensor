// png-to-ico ships no types for its internals. It has no `exports` map, so the
// subpath resolves; this is the one private path this package reaches into, to
// avoid carrying a second image resizer.
declare module 'png-to-ico/lib/png' {
    import type { PNG } from 'pngjs';
    export function resize(source: PNG, width: number, height: number): PNG;
}
