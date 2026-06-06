import * as vscode from 'vscode';
import { parseTensorJson, type TensorJson } from 'ptensor-ts';

/** Dtype names emitted by `p10::to_json_debug` (mirror of the view DTypeString). */
const KNOWN_DTYPES = new Set([
  'float32', 'float64', 'float16', 'uint8', 'uint16',
  'uint32', 'int8', 'int16', 'int32', 'int64',
]);

/** A named tensor ready to post to the viewer webview. `json` is the raw
 * debugger output (base64 blob); the tensor-view module decodes it. */
export interface NamedTensorJson {
  name: string;
  json: TensorJson;
}

const MISSING_HELPER_MESSAGE =
  "the debugger can't find p10::to_json_debug — it isn't linked into the " +
  'debuggee. Reference it once in your program (e.g. call ' +
  'p10::to_json_debug(some_tensor) so the linker keeps the symbol, then rebuild ' +
  'and restart the debug session.';

/** Debugger phrasings for a failed name lookup (symbol/identifier not found). */
function isLookupFailure(m: string): boolean {
  return (
    m.includes('no type named') ||
    m.includes('undeclared identifier') ||
    m.includes('use of undeclared') ||
    m.includes('no symbol') ||
    m.includes("couldn't look up symbols") ||
    m.includes('not found')
  );
}

/**
 * Maps a raw debugger evaluate error into a friendlier explanation, or returns
 * undefined to let the generic message through. Handles three common cases:
 * the `to_json_debug` helper not being linked, the expression not being in
 * scope, and a null / invalid dereference during evaluation.
 */
function explainEvalError(message: string, expression: string): string | undefined {
  const m = message.toLowerCase();

  // Helper symbol absent from the binary (references to_json_debug / p10).
  if ((m.includes('to_json_debug') || m.includes("'p10'")) && isLookupFailure(m)) {
    return MISSING_HELPER_MESSAGE;
  }

  // Expression not defined in the current frame.
  if (isLookupFailure(m) || m.includes('no member named') || m.includes('no variable named')) {
    return (
      `'${expression}' isn't available in the current frame — not in scope yet, ` +
      'optimized out, or misspelled. Step to where it is live and try again.'
    );
  }

  // Null pointer / invalid memory hit while evaluating (the eval crashed).
  if (
    m.includes('sigsegv') ||
    m.includes('exc_bad_access') ||
    m.includes('bad_access') ||
    m.includes('was interrupted') ||
    m.includes("couldn't apply expression side effects")
  ) {
    return `Couldn't read '${expression}' — it may be null or point to invalid memory.`;
  }

  return undefined;
}

/**
 * Reads the full NUL-terminated string that an evaluate result points at,
 * bypassing LLDB's summary truncation. `evalResult` is the raw evaluate text,
 * e.g. `(const char *) $0 = 0x000... "{...}"`; we pull the pointer out of it,
 * ask the debugger for the string length, then `readMemory` that many bytes.
 * Returns undefined if the adapter can't service the request so the caller can
 * fall back to the summary.
 */
async function readCStringFromMemory(
  session: vscode.DebugSession,
  frameId: number,
  evalResult: string
): Promise<string | undefined> {
  const ptrMatch = evalResult.match(/0x[0-9a-fA-F]+/);
  if (!ptrMatch) {
    return undefined;
  }
  const pointer = ptrMatch[0];
  if (/^0x0+$/.test(pointer)) {
    return undefined; // null pointer
  }

  try {
    const lenResp = await session.customRequest('evaluate', {
      expression: `(unsigned long long)strlen((const char *)${pointer})`,
      frameId,
      context: 'repl',
    });
    // Result looks like `(unsigned long long) $31 = 49152`; take the value after
    // `=`, not the `$31` temporary index.
    const length = Number((lenResp?.result as string | undefined)?.match(/=\s*(\d+)/)?.[1]);
    if (!Number.isFinite(length) || length <= 0) {
      return undefined;
    }

    // readMemory can return fewer bytes than asked, so loop until we have all.
    const chunks: Buffer[] = [];
    let offset = 0;
    while (offset < length) {
      const memResp = await session.customRequest('readMemory', {
        memoryReference: pointer,
        offset,
        count: length - offset,
      });
      const data = memResp?.data as string | undefined;
      if (!data) {
        return undefined;
      }
      const buf = Buffer.from(data, 'base64');
      if (buf.length === 0) {
        return undefined;
      }
      chunks.push(buf);
      offset += buf.length;
    }
    return Buffer.concat(chunks).toString('utf8');
  } catch {
    return undefined;
  }
}

export async function readTensor(
  session: vscode.DebugSession,
  frameId: number,
  expression: string
): Promise<NamedTensorJson> {
  const config = vscode.workspace.getConfiguration('ptensor');
  const maxBytes = config.get<number>('maxBytes', 64 * 1024 * 1024);

  let resp: { result?: unknown };
  try {
    resp = await session.customRequest('evaluate', {
      expression: `p10::to_json_debug(${expression})`,
      frameId,
      context: 'repl',
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(explainEvalError(msg, expression) ?? `Failed to evaluate ${expression}: ${msg}`);
  }
  if (typeof resp?.result !== 'string') {
    throw new Error(`evaluate returned no result for p10::to_json_debug(${expression}).`);
  }
  // A null/empty helper result (e.g. the expression is a null pointer) yields no
  // JSON object; parseTensorJson would fail with a cryptic message otherwise.
  if (!resp.result.includes('{')) {
    throw new Error(
      `Couldn't read '${expression}' — it may be null or not a Tensor (debugger returned: ${resp.result.slice(0, 80)}).`
    );
  }

  // The evaluate `result` is only LLDB's string *summary* of the returned
  // `const char*`, which the debugger truncates (target.max-string-summary-length)
  // and so cuts the JSON mid-blob. Read the NUL-terminated buffer straight from
  // process memory instead, which has no such cap. Fall back to the (possibly
  // truncated) summary if the memory path is unavailable.
  const rawJson = (await readCStringFromMemory(session, frameId, resp.result)) ?? resp.result;

  const json = parseTensorJson(rawJson);
  if (!KNOWN_DTYPES.has(json.dtype)) {
    throw new Error(`Unknown dtype '${json.dtype}' in tensor JSON.`);
  }

  const byteCount = Buffer.byteLength(json.blob, 'base64');
  if (byteCount === 0) {
    throw new Error('Tensor is empty (0 bytes).');
  }
  if (byteCount > maxBytes) {
    throw new Error(`Tensor is ${byteCount} bytes which exceeds ptensor.maxBytes=${maxBytes}.`);
  }

  return { name: expression, json };
}
