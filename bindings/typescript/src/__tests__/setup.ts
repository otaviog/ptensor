/**
 * Bun test preload: resolves the native C library path and sets
 * PTENSOR_LIB_PATH so that ffi.ts can find it at dlopen time.
 *
 * Checks the debug build tree first, then falls back to the system
 * library search path (i.e. env var already set by the caller).
 */
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

if (!process.env.PTENSOR_LIB_PATH) {
  const suffix = process.platform === 'darwin' ? 'dylib' : 'so';
  const candidate = resolve(
    import.meta.dir,
    '../../../../build/clang/debug/src/c',
    `libptensor_capi.${suffix}`,
  );
  if (existsSync(candidate)) {
    process.env.PTENSOR_LIB_PATH = candidate;
  }
}
