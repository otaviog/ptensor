# Ptensor TypeScript Bindings

TypeScript/JavaScript bindings for the Ptensor C API, providing a simple interface for tensor operations.

## Installation

```bash
npm install @ptensor/typescript
```

## Prerequisites

- Node.js 18 or higher
- The Ptensor native library must be built and available in your system

## Building from Source

1. Build the native Ptensor library first:

```bash
# From the project root
cmake --workflow --preset win-dev  # Windows
# or
cmake --workflow --preset lin-dev  # Linux/macOS
```

2. Install dependencies and build TypeScript bindings:

```bash
cd bindings/typescript
npm install
npm run build
```

## Usage

### Basic Example

```typescript
import { Tensor, DType } from '@ptensor/typescript';

// Create a tensor from an array
const data = new Float32Array([1, 2, 3, 4, 5, 6]);
const tensor = Tensor.fromData(data, [2, 3]);

console.log(tensor.toString());
// Output: Tensor(shape=[2, 3], dtype=FLOAT32, size=6)

console.log(tensor.shape);  // [2, 3]
console.log(tensor.ndim);   // 2
console.log(tensor.size);   // 6

// Get data back
const array = tensor.toArray();
console.log(array);  // [1, 2, 3, 4, 5, 6]

// Clean up
tensor.dispose();
```

### Creating Tensors

```typescript
// From TypedArray
const tensor1 = Tensor.fromData(
  new Float32Array([1, 2, 3, 4]),
  [2, 2]
);

// From plain array (will be converted to Float32Array by default)
const tensor2 = Tensor.fromData([1, 2, 3, 4], [2, 2]);

// With specific dtype
const tensor3 = Tensor.fromData(
  [1, 2, 3, 4],
  [2, 2],
  DType.INT32
);

// Zeros
const zeros = Tensor.zeros([3, 3], DType.FLOAT32);

// Ones
const ones = Tensor.ones([2, 4], DType.FLOAT64);
```

### Automatic Cleanup (Node.js 20+)

Using the `using` keyword for automatic resource management:

```typescript
function processData() {
  using tensor = Tensor.zeros([100, 100]);
  // Use tensor...
  // Automatically disposed when going out of scope
}
```

### Working with Different Data Types

```typescript
import { DType } from '@ptensor/typescript';

// Float types
const float32 = Tensor.zeros([2, 2], DType.FLOAT32);
const float64 = Tensor.zeros([2, 2], DType.FLOAT64);

// Integer types
const int32 = Tensor.zeros([2, 2], DType.INT32);
const uint8 = Tensor.zeros([2, 2], DType.UINT8);

// Get data as typed array
const data = tensor.getData<Float32Array>();
```

### Error Handling

```typescript
import { PtensorError } from '@ptensor/typescript';

try {
  const tensor = Tensor.fromData([1, 2, 3], [2, 2]); // Wrong size!
} catch (error) {
  if (error instanceof PtensorError) {
    console.error(`Ptensor error: ${error.code} - ${error.message}`);
  }
}
```

## API Reference

### Tensor

#### Static Methods

- `Tensor.fromData(data, shape, dtype?)` - Create tensor from data
- `Tensor.zeros(shape, dtype?)` - Create tensor filled with zeros
- `Tensor.ones(shape, dtype?)` - Create tensor filled with ones

#### Properties

- `dtype: DType` - Get the data type
- `shape: number[]` - Get the shape
- `ndim: number` - Get number of dimensions
- `size: number` - Get total number of elements

#### Methods

- `getData<T>(): T` - Get data as TypedArray
- `toArray(): number[]` - Get data as plain array
- `toString(): string` - Get string representation
- `dispose(): void` - Free native memory

### Enums

#### DType

- `FLOAT32`, `FLOAT64`, `FLOAT16`
- `UINT8`, `UINT16`, `UINT32`
- `INT8`, `INT16`, `INT32`, `INT64`

#### ErrorCode

- `OK`, `UNKNOWN_ERROR`, `ASSERTION_ERROR`
- `INVALID_ARGUMENT`, `INVALID_OPERATION`
- `OUT_OF_MEMORY`, `OUT_OF_RANGE`
- `NOT_IMPLEMENTED`, `OS_ERROR`, `IO_ERROR`

#### Device

- `CPU`, `CUDA`, `OCL`

## Environment Variables

- `PTENSOR_LIB_PATH` - Override the path to the native library

## Memory Management

Tensors wrap native C resources and must be explicitly disposed:

```typescript
const tensor = Tensor.zeros([100, 100]);
// ... use tensor ...
tensor.dispose(); // Important!
```

Or use automatic cleanup (Node.js 20+):

```typescript
using tensor = Tensor.zeros([100, 100]);
// Automatically disposed at end of scope
```

## License

MIT
