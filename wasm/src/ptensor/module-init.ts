import createP10, { P10 } from '../../build/p10.js';

export let MODULE: P10 | any = undefined;

/**
 * Initialize the P10 WebAssembly module
 */
export async function initP10(): Promise<void> {
  MODULE = await createP10();
}

export { P10 };