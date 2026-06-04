import * as vscode from 'vscode';
import { Dtype, dtypeName, NumericArray, TensorData } from './tensorReader';
import { computeStats, TensorStats } from './stats';

type ViewMode = 'table' | 'image' | 'large-table';

interface ImagePlane {
    width: number;
    height: number;
    channels: number;
    layout: 'interleaved' | 'planar';
}

interface ResolvedView {
    mode: ViewMode;
    image?: ImagePlane;
    /** Number of image tabs (N dim, or 1 for a single image). */
    batch: number;
}

export class TensorPanel {
    private static current: TensorPanel | undefined;
    private readonly panel: vscode.WebviewPanel;
    private disposables: vscode.Disposable[] = [];

    static show(_context: vscode.ExtensionContext, tensor: TensorData) {
        const column = vscode.window.activeTextEditor?.viewColumn ?? vscode.ViewColumn.Beside;
        if (TensorPanel.current) {
            TensorPanel.current.panel.reveal(column);
            TensorPanel.current.update(tensor);
            return;
        }
        const panel = vscode.window.createWebviewPanel(
            'ptensor.tensorView',
            `Tensor: ${tensor.expression}`,
            column,
            { enableScripts: true, retainContextWhenHidden: true }
        );
        TensorPanel.current = new TensorPanel(panel, tensor);
    }

    private constructor(panel: vscode.WebviewPanel, tensor: TensorData) {
        this.panel = panel;
        this.panel.onDidDispose(() => this.dispose(), null, this.disposables);
        this.update(tensor);
    }

    private update(tensor: TensorData) {
        this.panel.title = `Tensor: ${tensor.expression}`;
        this.panel.webview.html = renderHtml(tensor);
    }

    private dispose() {
        TensorPanel.current = undefined;
        this.panel.dispose();
        while (this.disposables.length) {
            this.disposables.pop()?.dispose();
        }
    }
}

function resolveView(shape: number[]): ResolvedView {
    const config = vscode.workspace.getConfiguration('ptensor');
    const threshold = config.get<number>('tableElementThreshold', 256);
    const total = shape.reduce((a, b) => a * b, 1);

    if (total <= threshold) {
        return { mode: 'table', batch: 1 };
    }

    // Recognise image-like layouts. Channels guessed by 1/3/4.
    const isChannel = (v: number) => v === 1 || v === 3 || v === 4;

    if (shape.length === 2) {
        const [h, w] = shape;
        return { mode: 'image', batch: 1, image: { height: h, width: w, channels: 1, layout: 'interleaved' } };
    }
    if (shape.length === 3) {
        // [H, W, C] or [C, H, W]
        const [a, b, c] = shape;
        if (isChannel(c)) {
            return {
                mode: 'image', batch: 1,
                image: { height: a, width: b, channels: c, layout: 'interleaved' },
            };
        }
        if (isChannel(a)) {
            return {
                mode: 'image', batch: 1,
                image: { height: b, width: c, channels: a, layout: 'planar' },
            };
        }
    }
    if (shape.length === 4) {
        // [N, C, H, W] or [N, H, W, C]
        const [n, a, b, c] = shape;
        if (isChannel(a)) {
            return {
                mode: 'image', batch: n,
                image: { height: b, width: c, channels: a, layout: 'planar' },
            };
        }
        if (isChannel(c)) {
            return {
                mode: 'image', batch: n,
                image: { height: a, width: b, channels: c, layout: 'interleaved' },
            };
        }
    }
    return { mode: 'large-table', batch: 1 };
}

function renderHtml(tensor: TensorData): string {
    const stats = computeStats(tensor.data);
    const view = resolveView(tensor.shape);

    let bodyHtml: string;
    if (view.mode === 'table') {
        bodyHtml = renderTable(tensor);
    } else if (view.mode === 'image' && view.image) {
        bodyHtml = renderImages(tensor, view.image, view.batch, stats);
    } else {
        bodyHtml = renderLargeTablePreview(tensor);
    }

    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  body { font-family: var(--vscode-font-family); color: var(--vscode-foreground); padding: 12px; }
  h2 { margin: 0 0 8px 0; font-size: 1.1em; }
  .meta { font-size: 0.9em; color: var(--vscode-descriptionForeground); margin-bottom: 12px; }
  .stats { display: flex; gap: 16px; margin-bottom: 12px; }
  .stats div { background: var(--vscode-editorWidget-background); padding: 4px 10px; border-radius: 4px; }
  .stats .label { color: var(--vscode-descriptionForeground); font-size: 0.8em; }
  table.tensor { border-collapse: collapse; font-family: var(--vscode-editor-font-family, monospace); font-size: 0.9em; }
  table.tensor td { padding: 2px 8px; border: 1px solid var(--vscode-panel-border); text-align: right; }
  .tabs { display: flex; gap: 4px; margin-bottom: 8px; flex-wrap: wrap; }
  .tabs button { background: var(--vscode-button-secondaryBackground); color: var(--vscode-button-secondaryForeground); border: none; padding: 4px 10px; cursor: pointer; }
  .tabs button.active { background: var(--vscode-button-background); color: var(--vscode-button-foreground); }
  canvas { image-rendering: pixelated; border: 1px solid var(--vscode-panel-border); max-width: 100%; }
  .pane { display: none; }
  .pane.active { display: block; }
  .warn { color: var(--vscode-editorWarning-foreground); font-style: italic; }
</style>
</head>
<body>
  <h2>${escapeHtml(tensor.expression)}</h2>
  <div class="meta">shape=[${tensor.shape.join(', ')}] dtype=${dtypeName(tensor.dtype)} bytes=${tensor.byteCount}</div>
  <div class="stats">
    <div><div class="label">min</div>${formatNumber(stats.min)}</div>
    <div><div class="label">max</div>${formatNumber(stats.max)}</div>
    <div><div class="label">mean</div>${formatNumber(stats.mean)}</div>
    <div><div class="label">count</div>${stats.count}</div>
  </div>
  ${bodyHtml}
</body>
</html>`;
}

function renderTable(tensor: TensorData): string {
    const flat = numericArrayToArray(tensor.data);
    const shape = tensor.shape;
    // Render as a 2D table for the last two dims; collapse leading dims as row labels.
    if (shape.length === 0) {
        return `<table class="tensor"><tr><td>${formatNumber(flat[0])}</td></tr></table>`;
    }
    if (shape.length === 1) {
        const cells = flat.map(v => `<td>${formatNumber(v)}</td>`).join('');
        return `<table class="tensor"><tr>${cells}</tr></table>`;
    }
    const cols = shape[shape.length - 1];
    const rows = flat.length / cols;
    const rowsHtml: string[] = [];
    for (let r = 0; r < rows; r++) {
        const cells: string[] = [];
        for (let c = 0; c < cols; c++) {
            cells.push(`<td>${formatNumber(flat[r * cols + c])}</td>`);
        }
        rowsHtml.push(`<tr>${cells.join('')}</tr>`);
    }
    return `<table class="tensor">${rowsHtml.join('')}</table>`;
}

function renderLargeTablePreview(tensor: TensorData): string {
    const flat = numericArrayToArray(tensor.data);
    const preview = flat.slice(0, 256);
    const cells = preview.map(v => `<td>${formatNumber(v)}</td>`).join('');
    return `<p class="warn">Tensor too large for a full table and shape is not image-like — showing first 256 elements.</p>
<table class="tensor"><tr>${cells}</tr></table>`;
}

function renderImages(
    tensor: TensorData,
    plane: ImagePlane,
    batch: number,
    stats: TensorStats
): string {
    const tabs: string[] = [];
    const panes: string[] = [];
    const planeElements = plane.width * plane.height * plane.channels;

    // Use min/max for float-ish data; uint8 maps directly.
    const useStretch = tensor.dtype !== Dtype.Uint8;
    const lo = useStretch ? stats.min : 0;
    const hi = useStretch ? stats.max : 255;
    const range = hi - lo;
    const scale = range > 0 ? 255 / range : 1;

    const flat = numericArrayToArray(tensor.data);
    for (let n = 0; n < batch; n++) {
        const start = n * planeElements;
        const rgba = planeToRgba(flat, start, plane, lo, scale);
        const dataUri = rgbaToPngDataUri(rgba, plane.width, plane.height);
        tabs.push(
            `<button class="tab-btn ${n === 0 ? 'active' : ''}" data-tab="${n}">image ${n}</button>`
        );
        panes.push(
            `<div class="pane ${n === 0 ? 'active' : ''}" data-pane="${n}">
                <canvas width="${plane.width}" height="${plane.height}"></canvas>
                <script>
                    (function(){
                        const c = document.querySelectorAll('canvas')[${n}];
                        const img = new Image();
                        img.onload = () => c.getContext('2d').drawImage(img, 0, 0);
                        img.src = "${dataUri}";
                    })();
                </script>
            </div>`
        );
    }

    const tabBar = batch > 1 ? `<div class="tabs">${tabs.join('')}</div>` : '';
    const tabScript = batch > 1 ? `
<script>
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const id = btn.getAttribute('data-tab');
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b === btn));
      document.querySelectorAll('.pane').forEach(p => p.classList.toggle('active', p.getAttribute('data-pane') === id));
    });
  });
</script>` : '';

    return `${tabBar}${panes.join('')}${tabScript}`;
}

function planeToRgba(
    flat: number[],
    offset: number,
    plane: ImagePlane,
    lo: number,
    scale: number
): Uint8ClampedArray {
    const { width, height, channels, layout } = plane;
    const rgba = new Uint8ClampedArray(width * height * 4);
    const planeSize = width * height;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            const getChan = (c: number): number => {
                if (layout === 'interleaved') {
                    return flat[offset + (y * width + x) * channels + c];
                }
                return flat[offset + c * planeSize + y * width + x];
            };
            const mapVal = (v: number) => Math.round((v - lo) * scale);
            if (channels === 1) {
                const v = mapVal(getChan(0));
                rgba[idx] = v;
                rgba[idx + 1] = v;
                rgba[idx + 2] = v;
                rgba[idx + 3] = 255;
            } else if (channels === 3) {
                rgba[idx] = mapVal(getChan(0));
                rgba[idx + 1] = mapVal(getChan(1));
                rgba[idx + 2] = mapVal(getChan(2));
                rgba[idx + 3] = 255;
            } else {
                rgba[idx] = mapVal(getChan(0));
                rgba[idx + 1] = mapVal(getChan(1));
                rgba[idx + 2] = mapVal(getChan(2));
                rgba[idx + 3] = mapVal(getChan(3));
            }
        }
    }
    return rgba;
}

function rgbaToPngDataUri(rgba: Uint8ClampedArray, width: number, height: number): string {
    // Use a minimal uncompressed PNG (with zlib stored blocks) so we don't pull dependencies.
    const png = buildPng(rgba, width, height);
    const base64 = Buffer.from(png).toString('base64');
    return `data:image/png;base64,${base64}`;
}

/**
 * Build a PNG with an uncompressed zlib stream (stored blocks). Small but lets us avoid
 * a zlib dependency in the webview.
 */
function buildPng(rgba: Uint8ClampedArray, width: number, height: number): Uint8Array {
    const sig = Uint8Array.of(137, 80, 78, 71, 13, 10, 26, 10);
    const ihdr = makeChunk('IHDR', (() => {
        const b = new Uint8Array(13);
        const v = new DataView(b.buffer);
        v.setUint32(0, width);
        v.setUint32(4, height);
        b[8] = 8;     // bit depth
        b[9] = 6;     // color type: RGBA
        b[10] = 0;    // compression
        b[11] = 0;    // filter
        b[12] = 0;    // interlace
        return b;
    })());

    // Filter byte (0) per scanline, prepended.
    const stride = width * 4;
    const raw = new Uint8Array((stride + 1) * height);
    for (let y = 0; y < height; y++) {
        raw[y * (stride + 1)] = 0;
        raw.set(rgba.subarray(y * stride, (y + 1) * stride), y * (stride + 1) + 1);
    }

    const compressed = zlibStored(raw);
    const idat = makeChunk('IDAT', compressed);
    const iend = makeChunk('IEND', new Uint8Array(0));

    const out = new Uint8Array(sig.length + ihdr.length + idat.length + iend.length);
    let off = 0;
    out.set(sig, off); off += sig.length;
    out.set(ihdr, off); off += ihdr.length;
    out.set(idat, off); off += idat.length;
    out.set(iend, off);
    return out;
}

function makeChunk(type: string, data: Uint8Array): Uint8Array {
    const typeBytes = new TextEncoder().encode(type);
    const length = data.length;
    const out = new Uint8Array(8 + length + 4);
    const v = new DataView(out.buffer);
    v.setUint32(0, length);
    out.set(typeBytes, 4);
    out.set(data, 8);
    const crc = crc32(out.subarray(4, 8 + length));
    v.setUint32(8 + length, crc);
    return out;
}

function zlibStored(data: Uint8Array): Uint8Array {
    // zlib header for default compression, no preset dict
    const header = Uint8Array.of(0x78, 0x01);
    // Split into max-65535 stored blocks
    const chunks: Uint8Array[] = [];
    const MAX = 65535;
    for (let i = 0; i < data.length; i += MAX) {
        const len = Math.min(MAX, data.length - i);
        const last = i + len >= data.length ? 1 : 0;
        const block = new Uint8Array(5 + len);
        block[0] = last;
        block[1] = len & 0xff;
        block[2] = (len >>> 8) & 0xff;
        block[3] = (~len) & 0xff;
        block[4] = ((~len) >>> 8) & 0xff;
        block.set(data.subarray(i, i + len), 5);
        chunks.push(block);
    }
    if (chunks.length === 0) {
        // Empty data still needs a terminator block.
        chunks.push(Uint8Array.of(1, 0, 0, 0xff, 0xff));
    }
    const adler = adler32(data);
    const tail = new Uint8Array(4);
    new DataView(tail.buffer).setUint32(0, adler);
    const total = chunks.reduce((s, c) => s + c.length, 0);
    const out = new Uint8Array(header.length + total + tail.length);
    let off = 0;
    out.set(header, off); off += header.length;
    for (const c of chunks) { out.set(c, off); off += c.length; }
    out.set(tail, off);
    return out;
}

function adler32(data: Uint8Array): number {
    let a = 1, b = 0;
    const M = 65521;
    for (let i = 0; i < data.length; i++) {
        a = (a + data[i]) % M;
        b = (b + a) % M;
    }
    return ((b << 16) | a) >>> 0;
}

let CRC_TABLE: Uint32Array | undefined;
function crc32(data: Uint8Array): number {
    if (!CRC_TABLE) {
        CRC_TABLE = new Uint32Array(256);
        for (let n = 0; n < 256; n++) {
            let c = n;
            for (let k = 0; k < 8; k++) {
                c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
            }
            CRC_TABLE[n] = c;
        }
    }
    let c = 0xffffffff;
    for (let i = 0; i < data.length; i++) {
        c = CRC_TABLE[(c ^ data[i]) & 0xff] ^ (c >>> 8);
    }
    return (c ^ 0xffffffff) >>> 0;
}

function numericArrayToArray(data: NumericArray): number[] {
    if (data instanceof BigInt64Array) {
        const out = new Array<number>(data.length);
        for (let i = 0; i < data.length; i++) {
            out[i] = Number(data[i]);
        }
        return out;
    }
    return Array.from(data as ArrayLike<number>);
}

function formatNumber(v: number): string {
    if (!Number.isFinite(v)) {
        return String(v);
    }
    if (Number.isInteger(v)) {
        return v.toString();
    }
    return v.toPrecision(6);
}

function escapeHtml(s: string): string {
    return s
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}
