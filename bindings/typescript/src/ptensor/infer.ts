import { ffiInt, ffiU64, newHandleBuf, readHandle } from './_internal';
import {
  p10_infer_destroy,
  p10_infer_from_onnx,
  p10_infer_get_input_count,
  p10_infer_get_output_count,
  p10_infer_run,
} from './backends/bun/ffi';
import { P10Error } from './p10Error';
import { _getRawHandle, _wrapHandle, type Tensor } from './tensor';

export type { Tensor };

export interface InferSession {
  /** Number of inputs the model expects. */
  getInputCount(): number;
  /** Number of outputs the model produces. */
  getOutputCount(): number;
  /**
   * Runs inference.
   * @param inputs - one Tensor per model input
   * @returns newly allocated output Tensors; caller must call delete() on each
   */
  run(inputs: Tensor[]): Tensor[];
  /** Releases the native session handle. */
  delete(): void;
}

class InferSessionImpl implements InferSession {
  // [0] = opaque P10Infer handle value
  private _buf: BigUint64Array;

  constructor(buf: BigUint64Array) {
    this._buf = buf;
  }

  getInputCount(): number {
    return Number(ffiU64(p10_infer_get_input_count(readHandle(this._buf))));
  }

  getOutputCount(): number {
    return Number(ffiU64(p10_infer_get_output_count(readHandle(this._buf))));
  }

  run(inputs: Tensor[]): Tensor[] {
    const numIn = inputs.length;
    const numOut = this.getOutputCount();

    // Pack input Ptensor handle values into a contiguous BigUint64Array.
    // The C API receives this as `const Ptensor*` — an array of void* values.
    const inputPtrs = new BigUint64Array(numIn);
    for (let i = 0; i < numIn; i++) {
      inputPtrs[i] = _getRawHandle(inputs[i]);
    }

    // Allocate output slots; C will fill each with a new Ptensor value.
    const outputPtrs = new BigUint64Array(numOut);

    P10Error.check(
      ffiInt(p10_infer_run(readHandle(this._buf), inputPtrs, numIn, outputPtrs, numOut)),
    );

    // Wrap each output handle in a Tensor the caller owns.
    return Array.from({ length: numOut }, (_, i) => {
      const buf = newHandleBuf();
      buf[0] = outputPtrs[i];
      return _wrapHandle(buf);
    });
  }

  delete(): void {
    if (this._buf[0] !== 0n) {
      P10Error.check(ffiInt(p10_infer_destroy(this._buf)));
    }
  }
}

/**
 * Loads an ONNX model and returns an inference session.
 * Call delete() on the session when done.
 */
export function fromOnnx(modelPath: string): InferSession {
  const buf = newHandleBuf();
  const pathBuf = Buffer.from(`${modelPath}\0`);
  P10Error.check(ffiInt(p10_infer_from_onnx(buf, pathBuf)));
  return new InferSessionImpl(buf);
}
