import * as vscode from 'vscode';

export enum Dtype {
    Float32 = 0,
    Float64 = 1,
    Float16 = 2,
    Uint8 = 3,
    Uint16 = 4,
    Uint32 = 5,
    Int8 = 6,
    Int16 = 7,
    Int32 = 8,
    Int64 = 9,
}

export type NumericArray =
    | Float32Array
    | Float64Array
    | Uint8Array
    | Uint16Array
    | Uint32Array
    | Int8Array
    | Int16Array
    | Int32Array
    | BigInt64Array;

export interface TensorData {
    expression: string;
    shape: number[];
    dtype: Dtype;
    data: NumericArray;
    byteCount: number;
}

interface TensorJson {
    dtype: string;
    shape: number[];
    stride: number[];
    blob: string;
}

const DTYPE_NAMES: Record<Dtype, string> = {
    [Dtype.Float32]: 'Float32',
    [Dtype.Float64]: 'Float64',
    [Dtype.Float16]: 'Float16',
    [Dtype.Uint8]: 'Uint8',
    [Dtype.Uint16]: 'Uint16',
    [Dtype.Uint32]: 'Uint32',
    [Dtype.Int8]: 'Int8',
    [Dtype.Int16]: 'Int16',
    [Dtype.Int32]: 'Int32',
    [Dtype.Int64]: 'Int64',
};

const DTYPE_SIZES: Record<Dtype, number> = {
    [Dtype.Float32]: 4,
    [Dtype.Float64]: 8,
    [Dtype.Float16]: 2,
    [Dtype.Uint8]: 1,
    [Dtype.Uint16]: 2,
    [Dtype.Uint32]: 4,
    [Dtype.Int8]: 1,
    [Dtype.Int16]: 2,
    [Dtype.Int32]: 4,
    [Dtype.Int64]: 8,
};

const DTYPE_FROM_NAME: Record<string, Dtype> = {
    float32: Dtype.Float32,
    float64: Dtype.Float64,
    float16: Dtype.Float16,
    uint8: Dtype.Uint8,
    uint16: Dtype.Uint16,
    uint32: Dtype.Uint32,
    int8: Dtype.Int8,
    int16: Dtype.Int16,
    int32: Dtype.Int32,
    int64: Dtype.Int64,
};

export function dtypeName(d: Dtype): string {
    return DTYPE_NAMES[d] ?? `Unknown(${d})`;
}

export function dtypeSize(d: Dtype): number {
    return DTYPE_SIZES[d] ?? 0;
}

export async function readTensor(
    session: vscode.DebugSession,
    frameId: number,
    expression: string
): Promise<TensorData> {
    const config = vscode.workspace.getConfiguration('ptensor');
    const maxBytes = config.get<number>('maxBytes', 64 * 1024 * 1024);

    const resp = await session.customRequest('evaluate', {
        expression: `p10::to_json_debug(${expression})`,
        frameId,
        context: 'repl',
    });
    if (typeof resp?.result !== 'string') {
        throw new Error(`evaluate returned no result for p10::to_json_debug(${expression}).`);
    }

    const parsed = parseTensorJson(resp.result);

    const dtype = DTYPE_FROM_NAME[parsed.dtype];
    if (dtype === undefined) {
        throw new Error(`Unknown dtype '${parsed.dtype}' in tensor JSON.`);
    }

    const byteCount = Buffer.byteLength(parsed.blob, 'base64');
    if (byteCount > maxBytes) {
        throw new Error(
            `Tensor is ${byteCount} bytes which exceeds ptensor.maxBytes=${maxBytes}.`
        );
    }
    if (byteCount === 0) {
        throw new Error('Tensor is empty (0 bytes).');
    }

    const buf = Buffer.from(parsed.blob, 'base64');
    const data = bytesToTyped(buf, dtype);

    return { expression, shape: parsed.shape, dtype, data, byteCount };
}

/**
 * Extracts the JSON object from a debugger `evaluate` result string. Debuggers
 * typically wrap a `const char*` value as `0x... "actual content"` with inner
 * quotes escaped as `\"`. We slice from the first `{` to the last `}` and
 * unescape the inner quotes/backslashes.
 */
function parseTensorJson(rawResult: string): TensorJson {
    const start = rawResult.indexOf('{');
    const end = rawResult.lastIndexOf('}');
    if (start < 0 || end <= start) {
        throw new Error(
            `Could not find a JSON object in evaluate result: ${rawResult.slice(0, 200)}`
        );
    }
    const jsonText = rawResult
        .slice(start, end + 1)
        .replace(/\\"/g, '"')
        .replace(/\\\\/g, '\\');

    let obj: unknown;
    try {
        obj = JSON.parse(jsonText);
    } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        throw new Error(`Could not parse tensor JSON: ${msg}`);
    }
    if (!obj || typeof obj !== 'object') {
        throw new Error('Tensor JSON is not an object.');
    }
    const o = obj as Record<string, unknown>;
    if (typeof o.dtype !== 'string' || !Array.isArray(o.shape) ||
        !Array.isArray(o.stride) || typeof o.blob !== 'string') {
        throw new Error('Tensor JSON is missing expected fields.');
    }
    return {
        dtype: o.dtype,
        shape: (o.shape as unknown[]).map(n => Number(n)),
        stride: (o.stride as unknown[]).map(n => Number(n)),
        blob: o.blob,
    };
}

function bytesToTyped(buf: Buffer, dtype: Dtype): NumericArray {
    const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
    switch (dtype) {
        case Dtype.Float32: return new Float32Array(ab);
        case Dtype.Float64: return new Float64Array(ab);
        case Dtype.Uint8:   return new Uint8Array(ab);
        case Dtype.Uint16:  return new Uint16Array(ab);
        case Dtype.Uint32:  return new Uint32Array(ab);
        case Dtype.Int8:    return new Int8Array(ab);
        case Dtype.Int16:   return new Int16Array(ab);
        case Dtype.Int32:   return new Int32Array(ab);
        case Dtype.Int64:   return new BigInt64Array(ab);
        case Dtype.Float16: return float16ToFloat32(new Uint16Array(ab));
    }
}

function float16ToFloat32(input: Uint16Array): Float32Array {
    const out = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) {
        const h = input[i];
        const sign = (h & 0x8000) >> 15;
        const exp = (h & 0x7c00) >> 10;
        const frac = h & 0x03ff;
        let val: number;
        if (exp === 0) {
            val = (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024);
        } else if (exp === 31) {
            val = frac ? NaN : (sign ? -Infinity : Infinity);
        } else {
            val = (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
        }
        out[i] = val;
    }
    return out;
}
