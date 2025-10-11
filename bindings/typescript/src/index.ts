/**
 * TypeScript bindings for Ptensor C API
 *
 * This module provides a high-level TypeScript interface to the Ptensor
 * tensor library through FFI bindings.
 *
 * @example
 * ```typescript
 * import { Tensor, DType } from '@ptensor/typescript';
 *
 * // Create a tensor from data
 * const tensor = Tensor.fromData(
 *   new Float32Array([1, 2, 3, 4, 5, 6]),
 *   [2, 3]
 * );
 *
 * console.log(tensor.toString());
 * console.log(tensor.shape); // [2, 3]
 * console.log(tensor.toArray()); // [1, 2, 3, 4, 5, 6]
 *
 * // Clean up
 * tensor.dispose();
 *
 * // Or use automatic cleanup (Node.js 20+)
 * using tensor2 = Tensor.zeros([3, 3], DType.FLOAT32);
 * ```
 */

export { Tensor } from './tensor';
export { DType, ErrorCode, Device } from './enums';
export { PtensorError } from './errors';
export type { TypedArrayLike } from './tensor';
