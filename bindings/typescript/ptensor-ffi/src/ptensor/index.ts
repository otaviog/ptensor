// Materialized, JS-owned tensor form and helpers (re-exported from ptensor-ts).
export { type NumericArray, parse, type Tensor, type TensorJson } from 'ptensor-ts';
export type {
  AudioFrame,
  MediaCapture,
  MediaTime,
  MediaWriter,
  Rational,
  VideoFrame,
} from './media';
export { createAudioFrame, createVideoFrame, openCapture, openWriter } from './media';
export { P10Error, P10ErrorCode } from './p10Error';
export type { DTypeString, PTensor, TypedArrayType } from './tensor';
export { fromArray, zeros } from './tensor';
