import { afterEach, describe, expect, it } from 'bun:test';
import { resolve } from 'node:path';
import { fromOnnx, type InferSession } from '../infer';
import { P10Error } from '../p10Error';
import { fromArray, type Tensor } from '../tensor';

// MNIST-12: input [1,1,28,28] float32 → output [1,10] float32
const MNIST_MODEL = resolve(import.meta.dir, '../../../../../src/infer/tests/data/mnist-12.onnx');

describe('InferSession integration (real C library)', () => {
  const sessions: InferSession[] = [];
  const tensors: Tensor[] = [];

  function trackSession(s: InferSession): InferSession {
    sessions.push(s);
    return s;
  }
  function trackTensor(t: Tensor): Tensor {
    tensors.push(t);
    return t;
  }

  afterEach(() => {
    for (const t of tensors.splice(0)) t.delete();
    for (const s of sessions.splice(0)) s.delete();
  });

  // ---------------------------------------------------------------- //
  // Session lifecycle
  // ---------------------------------------------------------------- //

  it('fromOnnx loads a valid model', () => {
    const session = trackSession(fromOnnx(MNIST_MODEL));
    expect(session).toBeDefined();
  });

  it('fromOnnx throws P10Error for a non-existent model', () => {
    expect(() => fromOnnx('/does/not/exist.onnx')).toThrow(P10Error);
  });

  it('delete() can be called safely', () => {
    const session = fromOnnx(MNIST_MODEL);
    expect(() => session.delete()).not.toThrow();
  });

  // ---------------------------------------------------------------- //
  // Model metadata
  // ---------------------------------------------------------------- //

  it('MNIST has 1 input', () => {
    const session = trackSession(fromOnnx(MNIST_MODEL));
    expect(session.getInputCount()).toBe(1);
  });

  it('MNIST has 1 output', () => {
    const session = trackSession(fromOnnx(MNIST_MODEL));
    expect(session.getOutputCount()).toBe(1);
  });

  // ---------------------------------------------------------------- //
  // Inference
  // ---------------------------------------------------------------- //

  it('run() with zeros input produces [1,10] float32 output', () => {
    const session = trackSession(fromOnnx(MNIST_MODEL));

    // MNIST expects [1, 1, 28, 28] float32
    const input = trackTensor(fromArray(new Float32Array(1 * 1 * 28 * 28), [1, 1, 28, 28]));
    const outputs = session.run([input]);
    for (const t of outputs) tensors.push(t);

    expect(outputs).toHaveLength(1);
    const out = outputs[0];
    expect(out.getDtype()).toBe('float32');
    expect(out.getShape()).toEqual([1n, 10n]);
    expect(out.getSize()).toBe(10n);
    expect(out.getSizeBytes()).toBe(40n); // 10 * 4
    expect(out.isEmpty()).toBe(false);
  });

  it('run() output tensors can be independently deleted', () => {
    const session = trackSession(fromOnnx(MNIST_MODEL));
    const input = trackTensor(fromArray(new Float32Array(1 * 1 * 28 * 28), [1, 1, 28, 28]));
    const outputs = session.run([input]);
    expect(() => {
      for (const t of outputs) t.delete();
    }).not.toThrow();
    // Don't push to tensors[] since already deleted.
  });

  it('run() throws P10Error when input count is wrong', () => {
    const session = trackSession(fromOnnx(MNIST_MODEL));
    expect(() => session.run([])).toThrow(P10Error);
  });

  it('run() can be called multiple times on the same session', () => {
    const session = trackSession(fromOnnx(MNIST_MODEL));
    const input = trackTensor(fromArray(new Float32Array(1 * 1 * 28 * 28), [1, 1, 28, 28]));

    const first = session.run([input]);
    const second = session.run([input]);

    for (const t of first) tensors.push(t);
    for (const t of second) tensors.push(t);

    expect(first[0].getShape()).toEqual([1n, 10n]);
    expect(second[0].getShape()).toEqual([1n, 10n]);
  });
});
