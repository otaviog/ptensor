import numpy as np
import pytest

from ptensor import Tensor, DType, P10Error, P10ErrorEnum, dtype_to_numpy, dtype_from_numpy


# ------------------------------------------------------------------ #
# dtype helpers
# ------------------------------------------------------------------ #


def test_dtype_to_numpy_all():
    assert dtype_to_numpy(DType.FLOAT32) == np.dtype("float32")
    assert dtype_to_numpy(DType.FLOAT64) == np.dtype("float64")
    assert dtype_to_numpy(DType.FLOAT16) == np.dtype("float16")
    assert dtype_to_numpy(DType.UINT8) == np.dtype("uint8")
    assert dtype_to_numpy(DType.UINT16) == np.dtype("uint16")
    assert dtype_to_numpy(DType.UINT32) == np.dtype("uint32")
    assert dtype_to_numpy(DType.INT8) == np.dtype("int8")
    assert dtype_to_numpy(DType.INT16) == np.dtype("int16")
    assert dtype_to_numpy(DType.INT32) == np.dtype("int32")
    assert dtype_to_numpy(DType.INT64) == np.dtype("int64")


def test_dtype_from_numpy_roundtrip():
    for np_dt in ["float32", "float64", "uint8", "uint16", "uint32", "int8", "int16", "int32", "int64"]:
        dt = dtype_from_numpy(np.dtype(np_dt))
        assert dtype_to_numpy(dt) == np.dtype(np_dt)


def test_dtype_from_numpy_unsupported():
    with pytest.raises(ValueError):
        dtype_from_numpy(np.dtype("complex64"))


# ------------------------------------------------------------------ #
# Tensor.from_numpy – basic construction
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "np_dtype",
    [np.float32, np.float64, np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.int64],
)
def test_from_numpy_dtype_roundtrip(np_dtype):
    data = np.arange(6, dtype=np_dtype).reshape(2, 3)
    tensor = Tensor.from_numpy(data)

    assert tensor.dtype == dtype_from_numpy(np.dtype(np_dtype))
    assert tensor.shape == (2, 3)
    assert tensor.ndim == 2
    assert tensor.size == 6
    assert not tensor.is_empty


def test_from_numpy_1d():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = Tensor.from_numpy(data)
    assert t.shape == (3,)
    assert t.ndim == 1
    assert t.size == 3


def test_from_numpy_3d():
    data = np.zeros((2, 3, 4), dtype=np.int32)
    t = Tensor.from_numpy(data)
    assert t.shape == (2, 3, 4)
    assert t.ndim == 3
    assert t.size == 24


def test_from_numpy_requires_ndarray():
    with pytest.raises(TypeError):
        Tensor.from_numpy([1, 2, 3])


# ------------------------------------------------------------------ #
# shape and strides
# ------------------------------------------------------------------ #


def test_shape_matches_numpy():
    data = np.zeros((5, 3, 7), dtype=np.float32)
    t = Tensor.from_numpy(data)
    assert t.shape == (5, 3, 7)


def test_contiguous_strides():
    # C-contiguous [2, 3] float32: strides in elements = (3, 1)
    data = np.zeros((2, 3), dtype=np.float32)
    t = Tensor.from_numpy(data)
    assert t.strides == (3, 1)


def test_noncontiguous_strides():
    # Transpose of [2, 3] gives shape [3, 2], strides in elements = (1, 3)
    data = np.zeros((2, 3), dtype=np.float32)
    t = Tensor.from_numpy(data.T)
    assert t.shape == (3, 2)
    assert t.strides == (1, 3)


# ------------------------------------------------------------------ #
# size_bytes
# ------------------------------------------------------------------ #


def test_size_bytes_float32():
    data = np.zeros((4, 4), dtype=np.float32)
    t = Tensor.from_numpy(data)
    assert t.size_bytes == 4 * 4 * 4  # 16 elements × 4 bytes


def test_size_bytes_int64():
    data = np.ones((3,), dtype=np.int64)
    t = Tensor.from_numpy(data)
    assert t.size_bytes == 3 * 8


# ------------------------------------------------------------------ #
# is_empty
# ------------------------------------------------------------------ #


def test_not_empty():
    data = np.array([1.0], dtype=np.float32)
    assert not Tensor.from_numpy(data).is_empty


# ------------------------------------------------------------------ #
# numpy() – data round-trip
# ------------------------------------------------------------------ #


def test_numpy_roundtrip_values():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32).reshape(2, 5)
    t = Tensor.from_numpy(data)
    result = t.numpy()
    np.testing.assert_array_equal(result, data)


def test_numpy_roundtrip_int32():
    data = np.array([[10, 20], [30, 40]], dtype=np.int32)
    t = Tensor.from_numpy(data)
    np.testing.assert_array_equal(t.numpy(), data)


def test_numpy_noncontiguous_values():
    # Transposed array: data is stored row-major but accessed column-major
    base = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    transposed = base.T  # shape (3, 2), non-contiguous
    t = Tensor.from_numpy(transposed)
    np.testing.assert_array_equal(t.numpy(), transposed)


def test_numpy_slice_values():
    base = np.arange(16, dtype=np.int32).reshape(4, 4)
    sliced = base[::2, ::2]  # shape (2, 2), non-contiguous
    t = Tensor.from_numpy(sliced)
    np.testing.assert_array_equal(t.numpy(), sliced)


# ------------------------------------------------------------------ #
# repr
# ------------------------------------------------------------------ #


def test_repr():
    data = np.zeros((2, 3), dtype=np.float32)
    t = Tensor.from_numpy(data)
    r = repr(t)
    assert "2" in r and "3" in r and "FLOAT32" in r
