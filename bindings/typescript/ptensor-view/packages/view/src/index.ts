// Public API of the tensor-view module: the React panel, and the tensor it
// takes. No native / FFI dependency is pulled in, so this entry is safe to
// consume from a VS Code webview, the Electron pilot, or a plain browser
// playground.
//
// What the panel is built out of -- `resolveView`, `computeStats`, the image
// and table components -- stays inside: it is how the panel decides what to
// draw, not something a host needs to reach.

export { TensorViewer } from './components/TensorViewer';
export type { TensorViewerProps } from './components/TensorViewer';
export { SampleBrowser } from './components/SampleBrowser';
export { SAMPLES } from './samples';

// The tensor and its transport form are ptensor-ts'; re-exported so a consumer
// of the panel has one import for the panel and what to feed it.
export type { Tensor, TensorJson } from 'ptensor-ts';
export { tensorFromJson } from 'ptensor-ts';
