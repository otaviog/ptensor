import * as vscode from 'vscode';
import { readTensor } from './readTensor';
import { TensorPanel } from './tensorPanel';
import { registerDebugTracker } from './debugTracker';

export function activate(context: vscode.ExtensionContext) {
    registerDebugTracker(context);

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
                const tensor = await loadTensor(expression);
                // Re-read from the *current* frame each time (the user may have
                // stepped since the panel opened).
                TensorPanel.show(context, tensor, () => loadTensor(expression));
            } catch (err) {
                const msg = err instanceof Error ? err.message : String(err);
                vscode.window.showErrorMessage(`ptensor: ${msg}`);
            }
        })
    );
}

/** Evaluates `expression` in the active debug session's current frame into a tensor. */
async function loadTensor(expression: string) {
    const session = vscode.debug.activeDebugSession;
    if (!session) {
        throw new Error('no active debug session.');
    }
    const frameId = await getActiveFrameId(session);
    if (frameId === undefined) {
        throw new Error('could not determine the active stack frame.');
    }
    return readTensor(session, frameId, expression);
}

export function deactivate() {}

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
