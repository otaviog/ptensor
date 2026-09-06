// Public API of the tensor-view module: the React panel and the utilities it
// renders with. No native / FFI dependency is pulled in, so this entry is safe
// to consume from a VS Code webview, the Electron pilot, or a plain browser
// playground.

export { TensorViewer } from './components/TensorViewer';
export type { TensorViewerProps } from './components/TensorViewer';
export { SampleBrowser } from './components/SampleBrowser';
export { SAMPLES } from './samples';

export { computeStats } from './stats';
export type { TensorStats } from './stats';

export { resolveView } from './resolveView';
export type { ImagePlane, ResolvedView, ViewMode } from './resolveView';

// The tensor, its transport form, its dtype vocabulary and the helpers for
// reading one are all ptensor-ts'; re-exported so a consumer of the panel has
// one import for the panel and everything it takes.
export type { DTypeString, NumericArray, Tensor, TensorJson } from 'ptensor-ts';
export { dtypeSizeBytes, elementAt, isFloatDtype, tensorFromJson } from 'ptensor-ts';
