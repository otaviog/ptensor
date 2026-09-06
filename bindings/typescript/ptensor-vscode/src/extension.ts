import * as vscode from 'vscode';
import { pushTensor } from './pushTensor';
import { TensorFeed } from './tensorFeed';
import { TensorPanel } from './tensorPanel';
import { registerDebugTracker } from './debugTracker';

let feed: TensorFeed | undefined;

export function activate(context: vscode.ExtensionContext) {
    registerDebugTracker(context);

    const channel = vscode.window.createOutputChannel('ptensor');
    context.subscriptions.push(channel);

    // Tensors arrive here, pushed by the debuggee, and land in the tab that
    // asked for them. One that nothing is waiting for is noted, not dropped
    // silently: it usually means the tab was closed mid-flight.
    feed = new TensorFeed({
        onTensor: (payload) => {
            channel.appendLine(`Received '${payload.name}' (${payload.tensor.dtype}).`);
            if (!TensorPanel.showTensor(payload)) {
                channel.appendLine(`No open tab for '${payload.name}', ignoring it.`);
            }
        },
        log: (message) => channel.appendLine(message),
    });
    context.subscriptions.push({ dispose: () => feed?.dispose() });

    context.subscriptions.push(
        vscode.commands.registerCommand('ptensor.previewSamples', () => {
            TensorPanel.showDemo(context);
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('ptensor.viewTensor', async (variable?: unknown) => {
            const expression = await resolveExpression(variable);
            if (!expression) {
                return;
            }
            try {
                // The tab opens first and waits: the tensor comes over the feed
                // once the debugger has made the call.
                TensorPanel.showPending(context, expression, () => requestTensor(expression));
                await requestTensor(expression);
            } catch (err) {
                const msg = err instanceof Error ? err.message : String(err);
                vscode.window.showErrorMessage(`ptensor: ${msg}`);
            }
        })
    );
}

/**
 * Asks the debuggee to push `expression` into our feed. Returns once the call
 * has been made; the tensor itself lands on the socket.
 */
async function requestTensor(expression: string): Promise<void> {
    const session = vscode.debug.activeDebugSession;
    if (!session) {
        throw new Error('no active debug session.');
    }
    const frameId = await getActiveFrameId(session);
    if (frameId === undefined) {
        throw new Error('could not determine the active stack frame.');
    }
    if (!feed) {
        throw new Error('the tensor feed is not running.');
    }
    await pushTensor(session, frameId, await feed.address(), expression);
}

export function deactivate() {
    feed?.dispose();
    feed = undefined;
}

async function resolveExpression(variable: unknown): Promise<string | undefined> {
    if (variable && typeof variable === 'object') {
        const v = variable as {
            evaluateName?: string;
            variable?: { evaluateName?: string; name?: string };
            name?: string;
        };
        const fromVar = v.evaluateName ?? v.variable?.evaluateName ?? v.variable?.name ?? v.name;
        if (fromVar) {
            return fromVar;
        }
    }
    return vscode.window.showInputBox({
        prompt: 'Tensor expression to visualize',
        placeHolder: 'e.g. my_tensor or *tensor_ptr',
    });
}

async function getActiveFrameId(session: vscode.DebugSession): Promise<number | undefined> {
    // Prefer the focused stack item (VS Code 1.89+).
    const active = (vscode.debug as unknown as { activeStackItem?: unknown }).activeStackItem;
    if (active && typeof active === 'object' && 'frameId' in (active as object)) {
        const frameId = (active as { frameId?: number }).frameId;
        if (typeof frameId === 'number') {
            return frameId;
        }
    }

    // Fall back to the first frame of the first thread that has any frames.
    try {
        const threadsResp = await session.customRequest('threads');
        const threads = threadsResp?.threads ?? [];
        for (const t of threads) {
            const st = await session.customRequest('stackTrace', {
                threadId: t.id,
                startFrame: 0,
                levels: 1,
            });
            if (st?.stackFrames?.length) {
                return st.stackFrames[0].id;
            }
        }
    } catch {
        // ignore, fall through
    }
    return undefined;
}
