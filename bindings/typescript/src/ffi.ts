import * as koffi from 'koffi';
import * as path from 'path';
import * as os from 'os';

/**
 * Find the native library based on the platform
 */
function findLibrary(): string {
  const platform = os.platform();
  const libName = platform === 'win32' ? 'ptensor_capi.dll' :
                  platform === 'darwin' ? 'libptensor_capi.dylib' :
                  'libptensor_capi.so';

  // Try multiple search paths
  const searchPaths = [
    // Relative to the bindings directory
    path.join(__dirname, '..', '..', '..', 'build', 'msbuild', 'native', 'c', 'Debug', libName),
    path.join(__dirname, '..', '..', '..', 'build', 'msbuild', 'native', 'c', 'Release', libName),
    path.join(__dirname, '..', '..', '..', 'build', 'lin-debug', 'native', 'c', libName),
    path.join(__dirname, '..', '..', '..', 'build', 'lin-release', 'native', 'c', libName),
    // System paths
    path.join('/usr', 'local', 'lib', libName),
    path.join('/usr', 'lib', libName),
    // Environment variable override
    process.env.PTENSOR_LIB_PATH || '',
  ];

  for (const libPath of searchPaths) {
    if (libPath && require('fs').existsSync(libPath)) {
      return libPath;
    }
  }

  // Fallback to just the library name (will use system search paths)
  return libName;
}

// Load the native library
const lib = koffi.load(findLibrary());

// Define C types
export const Ptensor = koffi.pointer('Ptensor', koffi.opaque());
export const PtensorPtr = koffi.out(koffi.pointer(Ptensor));

// Function signatures
export const p10_from_data = lib.func('p10_from_data', 'int', [
  PtensorPtr,           // Ptensor* tensor
  'int',                // P10DTypeEnum dtype
  koffi.pointer('int64_t'), // int64_t* shape
  'size_t',             // size_t num_dims
  koffi.pointer('uint8_t')  // uint8_t* data
]);

export const p10_destroy = lib.func('p10_destroy', 'int', [
  koffi.pointer(Ptensor)  // Ptensor* tensor
]);

export const p10_get_size = lib.func('p10_get_size', 'size_t', [
  Ptensor  // Ptensor tensor
]);

export const p10_get_dtype = lib.func('p10_get_dtype', 'int', [
  Ptensor  // Ptensor tensor
]);

export const p10_get_shape = lib.func('p10_get_shape', 'int', [
  Ptensor,                  // Ptensor tensor
  koffi.pointer('int64_t'), // int64_t* shape
  'size_t'                  // size_t num_dims
]);

export const p10_get_dimensions = lib.func('p10_get_dimensions', 'size_t', [
  Ptensor  // Ptensor tensor
]);

export const p10_get_data = lib.func('p10_get_data', koffi.pointer('void'), [
  Ptensor  // Ptensor tensor
]);

export const p10_get_last_error_message = lib.func('p10_get_last_error_message', 'string', []);
