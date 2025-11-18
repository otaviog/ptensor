/**
 * PTensor WebAssembly Example
 * Demonstrates basic usage of the PTensor TypeScript bindings
 */

import p10 from "./build/ptensor";

async function main() {
  console.log('Initializing P10 WebAssembly module...');
  await p10.init();
  console.log('Module loaded successfully!\n');

  const A = p10.zeros([3, 3], 'float32');
  console.log(A.wasmTensor.getShape().toString());

  return;
  // Example 1: Create a tensor from a Float32Array
  console.log('=== Example 1: Create from Float32Array ===');
  const data1 = new Float32Array([1, 2, 3, 4, 5, 6]);
  const tensor1 = Tensor.fromTypedArray(data1, [2, 3]);

  console.log('Tensor 1:');
  console.log('  Shape:', tensor1.shape);
  console.log('  DType:', tensor1.dtype);
  console.log('  Size:', tensor1.size);
  console.log('  String:', tensor1.toString());

  // Example 2: Create a zeros tensor
  console.log('\n=== Example 2: Create zeros tensor ===');
  const tensor2 = Tensor.zeros([3, 3], 'float32');
  console.log('Tensor 2 (zeros):');
  console.log('  Shape:', tensor2.shape);
  console.log('  DType:', tensor2.dtype);
  console.log('  Size:', tensor2.size);
  console.log('  String:', tensor2.toString());

  // Example 3: Create a tensor with uint8 data
  console.log('\n=== Example 3: Uint8 tensor ===');
  const data3 = new Uint8Array([10, 20, 30, 40]);
  const tensor3 = Tensor.fromTypedArray(data3, [2, 2]);
  console.log('Tensor 3 (uint8):');
  console.log('  Shape:', tensor3.shape);
  console.log('  DType:', tensor3.dtype);
  console.log('  Size:', tensor3.size);

  // Example 4: Different shapes
  console.log('\n=== Example 4: Different tensor shapes ===');
  const tensor4a = Tensor.zeros([10], 'float64');
  console.log('1D tensor [10]:', { shape: tensor4a.shape, dtype: tensor4a.dtype });

  const tensor4b = Tensor.zeros([2, 3, 4], 'float32');
  console.log('3D tensor [2, 3, 4]:', { shape: tensor4b.shape, dtype: tensor4b.dtype });

  // Example 5: Different data types
  console.log('\n=== Example 5: Different data types ===');
  const dtypes: Array<'float32' | 'float64' | 'int32' | 'uint8'> = ['float32', 'float64', 'int32', 'uint8'];
  const dtypeTensors = dtypes.map(dtype => {
    const t = Tensor.zeros([2, 2], dtype);
    console.log(`  ${dtype} tensor:`, { shape: t.shape, dtype: t.dtype, size: t.size });
    return t;
  });

  // Clean up - free memory
  console.log('\n=== Cleaning up ===');
  tensor1.delete();
  tensor2.delete();
  tensor3.delete();
  tensor4a.delete();
  tensor4b.delete();
  dtypeTensors.forEach(t => t.delete());

  console.log('All tensors cleaned up successfully!');
}

main().catch(error => {
  console.error('Error:', error);
  process.exit(1);
});
