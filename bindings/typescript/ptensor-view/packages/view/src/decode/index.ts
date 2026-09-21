// Decoding a tensor for display, off the calling thread where the host allows
// it. Imported as `@ptensor/tensor-view/decode`.
//
// A separate entry from the panel on purpose: this half pulls no React, so an
// extension host or a worker can use it without dragging the renderer in.

export { createDecoder, createInlineDecoder, type DecoderLogger, type TensorDecoder } from './decoder';
export {
    fetchTensorJson,
    fromTransfer,
    toTransfer,
    type DecodedTransfer,
    type DecodeRequest,
    type DecodeResponse,
} from './decodeTransfer';
