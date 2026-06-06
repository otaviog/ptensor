import * as vscode from 'vscode';

/**
 * Tags ptensor variables in the debugger's Variables view so the "Visualize
 * Tensor" context-menu item only appears on tensors.
 *
 * VS Code sets the `debugProtocolVariableMenuContext` context key from each DAP
 * Variable's non-standard `__vscodeVariableMenuContext` field. The stock debug
 * adapters (lldb-dap, cpptools) never set it, so we intercept the `variables`
 * responses on their way to the UI and stamp the field on anything whose type
 * looks like a `p10::Tensor` (value, reference, pointer, const-qualified).
 */
const MENU_CONTEXT = 'ptensorTensor';
const TENSOR_TYPE = /\bp10::Tensor\b/;

interface DapVariable {
    type?: string;
    __vscodeVariableMenuContext?: string;
}

interface DapMessage {
    type?: string;
    command?: string;
    body?: { variables?: DapVariable[] };
}

export function registerDebugTracker(context: vscode.ExtensionContext): void {
    context.subscriptions.push(
        vscode.debug.registerDebugAdapterTrackerFactory('*', {
            createDebugAdapterTracker() {
                return {
                    onDidSendMessage(message: DapMessage) {
                        if (message.type !== 'response' || message.command !== 'variables') {
                            return;
                        }
                        for (const variable of message.body?.variables ?? []) {
                            if (variable.type && TENSOR_TYPE.test(variable.type)) {
                                variable.__vscodeVariableMenuContext = MENU_CONTEXT;
                            }
                        }
                    },
                };
            },
        })
    );
}
