import * as vscode from 'vscode';
import { readTensor } from './tensorReader';
import { TensorPanel } from './tensorPanel';

export function activate(context: vscode.ExtensionContext) {
    context.subscriptions.push(
        vscode.commands.registerCommand('ptensor.viewTensor', async (variable?: unknown) => {
            const session = vscode.debug.activeDebugSession;
            if (!session) {
                vscode.window.showErrorMessage('ptensor: no active debug session.');
                return;
            }

            const expression = await resolveExpression(variable);
            if (!expression) {
                return;
            }

            const frameId = await getActiveFrameId(session);
            if (frameId === undefined) {
                vscode.window.showErrorMessage('ptensor: could not determine the active stack frame.');
                return;
            }

            try {
                const tensor = await readTensor(session, frameId, expression);
                TensorPanel.show(context, tensor);
            } catch (err) {
                const msg = err instanceof Error ? err.message : String(err);
                vscode.window.showErrorMessage(`ptensor: ${msg}`);
            }
        })
    );
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
