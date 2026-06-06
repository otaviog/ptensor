// Public API of the tensor-view module: the shared TensorView type, the
// React panel, transport helpers, and the lower-level utilities. No native /
// FFI dependency is pulled in here, so this entry is safe to consume from a
// VS Code webview, the Electron pilot, or a plain browser playground.

export type { DTypeString, NumericArray, TensorView } from './types';
export { DTYPE_SIZES, elementAt, isFloatDtype } from './types';

export { TensorViewer } from './components/TensorViewer';
export type { TensorViewerProps } from './components/TensorViewer';
export { SampleBrowser } from './components/SampleBrowser';
export { SAMPLES } from './samples';

export { computeStats } from './stats';
export type { TensorStats } from './stats';

export { resolveView } from './resolveView';
export type { ImagePlane, ResolvedView, ViewMode } from './resolveView';

export { base64ToArrayBuffer, bytesToTyped, fromTensorJson } from './tensorView';
// TensorJson is owned by ptensor-ts; re-exported here for the view's public API.
export type { TensorJson } from 'ptensor-ts';
