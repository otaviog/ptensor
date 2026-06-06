/**
 * The wire format emitted by `p10::to_json_debug` and by binaries that print a
 * tensor to stdout: `{dtype, shape, stride, blob}` where `blob` is the raw
 * little-endian element bytes, base64-encoded.
 */
export type TensorJson = {
  dtype: string;
  shape: number[];
  stride: number[];
  blob: string;
};

/**
 * Extracts and parses a `TensorJson` from a raw debugger / stdout string. The
 * input may be wrapped (e.g. an LLDB summary `(const char *) $0 = 0x.. "{...}"`),
 * so we slice between the first `{` and last `}` and undo C-string escaping.
 */
export function parseTensorJson(rawResult: string): TensorJson {
  const start = rawResult.indexOf('{');
  const end = rawResult.lastIndexOf('}');
  if (start < 0 || end <= start) {
    throw new Error(
      `Could not find a JSON object in evaluate result: ${rawResult.slice(0, 200)}`,
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
  if (
    typeof o.dtype !== 'string' ||
    !Array.isArray(o.shape) ||
    !Array.isArray(o.stride) ||
    typeof o.blob !== 'string'
  ) {
    throw new Error('Tensor JSON is missing expected fields.');
  }
  return {
    dtype: o.dtype,
    shape: (o.shape as unknown[]).map((n) => Number(n)),
    stride: (o.stride as unknown[]).map((n) => Number(n)),
    blob: o.blob,
  };
}
