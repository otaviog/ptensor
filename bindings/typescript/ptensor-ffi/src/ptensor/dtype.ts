// The dtype names and the C `P10DTypeEnum` mapping live in ptensor-ts so the
// FFI binding, the webview, and the VS Code extension share one definition.
export { type DTypeString, dtypeToNumber, numberToDtype } from 'ptensor-ts';
