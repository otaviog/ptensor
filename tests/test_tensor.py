import numpy as np

from ptensor import Tensor


def test_tensor_binding():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32).reshape(2, 5)
    tensor = Tensor.from_numpy(data)

    with tensor.numpy(always_convert=True) as np_data:
        assert np_data.tolist() == data.tolist()
