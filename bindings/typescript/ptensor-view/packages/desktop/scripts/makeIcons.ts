// Cuts the app icons from the one source in docs/, so this package keeps no
// icon of its own. Runs before every build (see the `dev` and `build` scripts);
// the output is derived and git-ignored.
//
//   bun run icons          # or --force to rebuild an up-to-date set
//
// Pure JS on purpose: `sips`/`iconutil` are macOS-only, and the Windows and
// Linux icons have to be cuttable on those platforms too. (`iconutil` still
// turns the .iconset into AppIcon.icns, but only macOS builds need that.)

import pngToIco from 'png-to-ico';
import { PNG } from 'pngjs';
import { mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const packageRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const SOURCE = join(packageRoot, '../../../../../docs/icon-light.png');
const OUT_DIR = join(packageRoot, '.icons');

/** macOS: `iconutil` compiles this folder into the bundle's AppIcon.icns. */
const ICONSET: ReadonlyArray<[size: number, name: string]> = [
    [16, 'icon_16x16'],
    [32, 'icon_16x16@2x'],
    [32, 'icon_32x32'],
    [64, 'icon_32x32@2x'],
    [128, 'icon_128x128'],
    [256, 'icon_128x128@2x'],
    [256, 'icon_256x256'],
    [512, 'icon_256x256@2x'],
    [512, 'icon_512x512'],
    [1024, 'icon_512x512@2x'],
];

/** Windows: the sizes that go into the multi-image .ico. */
const ICO_SIZES = [16, 24, 32, 48, 64, 128, 256];

/** Linux: a single PNG, copied into the bundle as appIcon.png. */
const LINUX_SIZE = 512;

/**
 * Centers the image on a transparent square canvas. png-to-ico refuses a
 * non-square source, and an .iconset entry has to be square -- the source is
 * 250x251, so without this every size would be stretched by a pixel.
 */
function square(source: PNG): PNG {
    const side = Math.max(source.width, source.height);
    if (side === source.width && side === source.height) {
        return source;
    }
    const out = new PNG({ width: side, height: side });
    const dx = Math.floor((side - source.width) / 2);
    const dy = Math.floor((side - source.height) / 2);
    PNG.bitblt(source, out, 0, 0, source.width, source.height, dx, dy);
    return out;
}

/** Bicubic, borrowed from png-to-ico so this script carries no resizer. */
async function resize(source: PNG, size: number): Promise<PNG> {
    const { resize: resizePng } = await import('png-to-ico/lib/png');
    return resizePng(source, size, size) as PNG;
}

/** True when every output is newer than the source, so there is nothing to do. */
function upToDate(outputs: string[]): boolean {
    let sourceTime: number;
    try {
        sourceTime = statSync(SOURCE).mtimeMs;
    } catch {
        return false;
    }
    return outputs.every((path) => {
        try {
            return statSync(path).mtimeMs >= sourceTime;
        } catch {
            return false;
        }
    });
}

const iconsetDir = join(OUT_DIR, 'icon.iconset');
const outputs = [
    ...ICONSET.map(([, name]) => join(iconsetDir, `${name}.png`)),
    join(OUT_DIR, 'icon.ico'),
    join(OUT_DIR, 'icon.png'),
];

if (!process.argv.includes('--force') && upToDate(outputs)) {
    console.log('icons: up to date');
    process.exit(0);
}

mkdirSync(iconsetDir, { recursive: true });

const source = square(PNG.sync.read(readFileSync(SOURCE)));
if (source.width < 1024) {
    console.warn(
        `icons: ${SOURCE} is ${source.width}px, so the sizes above that are upscaled`
    );
}

// One resize per distinct size, reused by the entries that share it.
const scaled = new Map<number, PNG>();
for (const size of new Set([...ICONSET.map(([s]) => s), ...ICO_SIZES, LINUX_SIZE])) {
    scaled.set(size, await resize(source, size));
}
const encoded = new Map(
    [...scaled].map(([size, png]) => [size, PNG.sync.write(png)] as const)
);

for (const [size, name] of ICONSET) {
    writeFileSync(join(iconsetDir, `${name}.png`), encoded.get(size)!);
}
writeFileSync(join(OUT_DIR, 'icon.png'), encoded.get(LINUX_SIZE)!);
writeFileSync(
    join(OUT_DIR, 'icon.ico'),
    new Uint8Array(await pngToIco(ICO_SIZES.map((size) => encoded.get(size)!)))
);

console.log(`icons: wrote ${outputs.length} files under ${OUT_DIR} from ${SOURCE}`);
