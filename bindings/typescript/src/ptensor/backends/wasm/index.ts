/**
 * WASM backend for ptensor-js.
 *
 * Usage:
 *   import { loadWasm } from './ptensor/backends/wasm';
 *
 *   const api = await loadWasm('/path/to/ptensor.js');
 *   const t = api.fromArray(new Float32Array([1, 2, 3]), [3]);
 *   console.log(t.getShape());   // [3n]
 *   console.log(t.getDtype());   // 'float32'
 *   t.delete();
 *
 * Note: media and infer APIs are not available in the WASM build.
 */

import type { DTypeString, Tensor, TypedArrayType } from '../../types';
import type { PTensorModuleFactory, PTensorWasmModule } from './module';
import { fromArray as _fromArray, zeros as _zeros } from './tensor';

export type { DTypeString, Tensor, TypedArrayType };

/** The public API surface exposed after loading the WASM module. */
export interface PtensorWasmApi {
  /**
   * Creates a Tensor view backed by a copy of `data` on the WASM heap.
   * Custom per-element strides (not byte strides) may optionally be supplied.
   * Call `delete()` when done.
   */
  fromArray(data: TypedArrayType, shape: number[], strides?: number[]): Tensor;

  /**
   * Creates a zero-filled Tensor of the given shape and dtype.
   * Call `delete()` when done.
   */
  zeros(shape: number[], dtype?: DTypeString): Tensor;
}

/**
 * Loads the PTensor WASM module and returns the tensor API.
 *
 * @param loaderOrFactory
 *   - A path / URL string to the Emscripten-generated `.js` loader file, or
 *   - A `PTensorModuleFactory` function (already `import()`-ed), or
 *   - A pre-instantiated `PTensorWasmModule` (for testing / custom loaders).
 *
 * @example
 * // Browser
 * const api = await loadWasm('/assets/ptensor.js');
 *
 * @example
 * // Node.js (dynamic import of generated loader)
 * const api = await loadWasm(new URL('./ptensor.js', import.meta.url).pathname);
 *
 * @example
 * // Provide a pre-built factory function
 * import PTensorModule from './ptensor.js';
 * const api = await loadWasm(PTensorModule);
 */
export async function loadWasm(
  loaderOrFactory: string | URL | PTensorModuleFactory | PTensorWasmModule,
): Promise<PtensorWasmApi> {
  let mod: PTensorWasmModule;

  if (isPreinstantiatedModule(loaderOrFactory)) {
    mod = loaderOrFactory;
  } else if (typeof loaderOrFactory === 'function') {
    mod = await (loaderOrFactory as PTensorModuleFactory)();
  } else {
    // string or URL — dynamic import the Emscripten JS loader
    const url = typeof loaderOrFactory === 'string' ? loaderOrFactory : loaderOrFactory.toString();
    const { default: factory } = await import(/* @vite-ignore */ url);
    mod = await (factory as PTensorModuleFactory)();
  }

  return buildApi(mod);
}

function buildApi(mod: PTensorWasmModule): PtensorWasmApi {
  return {
    fromArray: (data, shape, strides) => _fromArray(mod, data, shape, strides),
    zeros: (shape, dtype) => _zeros(mod, shape, dtype),
  };
}

/** Returns true when `v` looks like an already-instantiated Emscripten module. */
function isPreinstantiatedModule(v: unknown): v is PTensorWasmModule {
  return (
    typeof v === 'object' &&
    v !== null &&
    typeof (v as PTensorWasmModule)._malloc === 'function' &&
    typeof (v as PTensorWasmModule).stackSave === 'function'
  );
}
