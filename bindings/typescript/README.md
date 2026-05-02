# PTensor WebAssembly Bindings

WebAssembly bindings for PTensor, enabling tensor operations in browsers and Node.js.

## Building

Build the WebAssembly module from the project root:

```bash
cmake --workflow --preset wasm-build
```

This generates:
- `build/wasm-release/ptensor.js` - JavaScript loader
- `build/wasm-release/ptensor.wasm` - WebAssembly binary

Copy these files to this directory for use with the TypeScript bindings.

## Setup

Install dependencies:

```bash
cd wasm
npm install
```

Compile TypeScript:

```bash
npm run build
```

## Usage

### Basic Example

```typescript
import { initPTensor, Tensor } from './ptensor-wrapper';

async function main() {
  // Initialize the WebAssembly module
  const module = await initPTensor();

  // Create a tensor from a Float32Array
  const data = new Float32Array([1, 2, 3, 4, 5, 6]);
  const tensor = Tensor.fromTypedArray(module, data, [2, 3]);

  console.log('Shape:', tensor.shape);     // [2, 3]
  console.log('DType:', tensor.dtype);     // 'float32'
  console.log('Size:', tensor.size);       // 6
  console.log('NDim:', tensor.ndim);       // 2

  // Clean up
  tensor.delete();
}

main();
```

### Creating Tensors

```typescript
// From typed arrays
const tensor1 = Tensor.fromTypedArray(module, new Float32Array([1, 2, 3, 4]), [2, 2]);

// Zeros
const tensor2 = Tensor.zeros(module, [3, 3], 'float32');

// Ones
const tensor3 = Tensor.ones(module, [2, 2], 'int32');

// Different dtypes
const uint8Tensor = Tensor.fromTypedArray(module, new Uint8Array([10, 20, 30, 40]), [2, 2]);
```

### Supported Data Types

- `float32` - 32-bit floating point
- `float64` - 64-bit floating point
- `uint8` - 8-bit unsigned integer
- `uint16` - 16-bit unsigned integer
- `uint32` - 32-bit unsigned integer
- `int8` - 8-bit signed integer
- `int16` - 16-bit signed integer
- `int32` - 32-bit signed integer
- `int64` - 64-bit signed integer

### Memory Management

Always call `delete()` on tensors when done to free WebAssembly memory:

```typescript
const tensor = Tensor.zeros(module, [100, 100]);
// ... use tensor ...
tensor.delete();  // Important!
```

## Project Structure

- `module.cpp` - Emscripten bindings (C++)
- `wasm_shape.hpp` - Shape class wrapper for WebAssembly
- `wasm_tensor.hpp` - Tensor class wrapper for WebAssembly
- `ptensor.d.ts` - TypeScript type definitions for raw WASM bindings
- `ptensor-wrapper.ts` - High-level TypeScript API
- `example.ts` - Usage examples
- `package.json` - NPM package configuration
- `tsconfig.json` - TypeScript compiler configuration

## API Reference

### `initPTensor()`

Initialize the WebAssembly module. Must be called before creating tensors.

**Returns:** `Promise<PTensorModule>`

### `Tensor.fromTypedArray(module, data, shape)`

Create a tensor from a typed array.

**Parameters:**
- `module: PTensorModule` - The initialized WASM module
- `data: TypedArrayType` - Source data
- `shape: number[]` - Tensor dimensions

**Returns:** `Tensor`

### `Tensor.zeros(module, shape, dtype?)`

Create a tensor filled with zeros.

**Parameters:**
- `module: PTensorModule` - The initialized WASM module
- `shape: number[]` - Tensor dimensions
- `dtype?: DTypeString` - Data type (default: 'float32')

**Returns:** `Tensor`

### `Tensor.ones(module, shape, dtype?)`

Create a tensor filled with ones.

**Parameters:**
- `module: PTensorModule` - The initialized WASM module
- `shape: number[]` - Tensor dimensions
- `dtype?: DTypeString` - Data type (default: 'float32')

**Returns:** `Tensor`

### Tensor Properties

- `tensor.shape: number[]` - Get tensor dimensions
- `tensor.dtype: DTypeString` - Get data type
- `tensor.size: number` - Get total number of elements
- `tensor.ndim: number` - Get number of dimensions

### Tensor Methods

- `tensor.delete(): void` - Free WebAssembly memory

## Running Examples

```bash
npm run example
```

## Notes

- The TypeScript wrapper handles memory allocation and data copying between JavaScript and WebAssembly
- Shape and dtype are created using the bound C++ classes (`WasmShape`, `Dtype`)
- Some advanced features may require additional C++ bindings
