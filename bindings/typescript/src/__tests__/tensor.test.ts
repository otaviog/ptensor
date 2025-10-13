import { it, describe, expect } from 'vitest'
import { Tensor } from '../tensor';
import { DType } from '../enums';

describe('Tensor creating methods', () => {
  it('should be created from Float32Array', ()=> {
    const data = new Float32Array([1, 2, 3, 4, 5, 6]);
    const tensor = Tensor.fromData(data, [2, 3]);

    expect(tensor.ndim).toBe(2);
    expect(tensor.shape).toEqual([2, 3]);
    expect(tensor.dtype).toBe(DType.FLOAT32);
    expect(tensor.size).toBe(6);
  });
});
