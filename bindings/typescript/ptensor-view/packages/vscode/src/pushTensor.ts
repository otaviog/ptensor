// Asks the debuggee to push a tensor into the extension's feed.
//
// The whole transfer is one `evaluate` of `p10::tlog::log_to(...)` in the
// stopped frame: the debugger only has to make the call, and the tensor itself
// leaves over a socket. Nothing is read back through the debugger, so the size
// of the tensor no longer decides whether this works.

import * as vscode from 'vscode';

const MISSING_TLOG_MESSAGE =
    "the debugger can't find p10::tlog::log_to — it isn't linked into the " +
    'debuggee. Link ptensor_tlog and reference it once (e.g. call ' +
    'p10::tlog::log_to("127.0.0.1:1", "anchor", some_tensor)) so the linker ' +
    'keeps the symbol, then rebuild and restart the debug session.';

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
 * undefined to let the generic message through.
 */
function explainEvalError(message: string, expression: string): string | undefined {
    const m = message.toLowerCase();

    if ((m.includes('log_to') || m.includes('tlog') || m.includes("'p10'")) && isLookupFailure(m)) {
        return MISSING_TLOG_MESSAGE;
    }

    if (isLookupFailure(m) || m.includes('no member named') || m.includes('no variable named')) {
        return (
            `'${expression}' isn't available in the current frame — not in scope yet, ` +
            'optimized out, or misspelled. Step to where it is live and try again.'
        );
    }

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

/** C string literal for a value the extension controls (an address, a name). */
function quote(value: string): string {
    return `"${value.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`;
}

/**
 * Evaluates the log_to call. Resolves once the debugger has made the call --
 * the tensor arrives separately, on the feed's socket.
 */
export async function pushTensor(
    session: vscode.DebugSession,
    frameId: number,
    address: string,
    expression: string
): Promise<void> {
    const call = `p10::tlog::log_to(${quote(address)}, ${quote(expression)}, ${expression})`;
    try {
        await session.customRequest('evaluate', {
            expression: call,
            frameId,
            context: 'repl',
        });
    } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        throw new Error(explainEvalError(msg, expression) ?? `Failed to evaluate ${call}: ${msg}`);
    }
}
