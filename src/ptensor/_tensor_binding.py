from ctypes import (
    c_int,
    c_void_p,
    POINTER,
    c_size_t,
    c_uint8,
    c_float,
    c_int64,
    cast,
)
from contextlib import contextmanager
from typing import Optional, Generator

import numpy as np

from ._core_binding import get_library, P10ErrorEnum

P10Tensor = c_void_p


class DType(c_int):
    FLOAT32 = 0
    UINT8 = 1
    INT64 = 2

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, DType):
            return self.value == other.value
        return self.value == other

    def __ne__(self, other):
        if isinstance(other, DType):
            return self.value != other.value
        return self.value != other

    def __hash__(self):
        return hash(self.value)


_LIB = get_library()

######
## p10_tensor_from_data
_LIB.p10_tensor_from_data.argtypes = [
    POINTER(P10Tensor),
    DType,
    POINTER(c_int64),
    c_size_t,
    POINTER(c_uint8),
]
_LIB.p10_tensor_from_data.restype = P10ErrorEnum

######
## p10_tensor_destroy
_LIB.p10_tensor_destroy.argtypes = [POINTER(P10Tensor)]
_LIB.p10_tensor_destroy.restype = c_int

######
## p10_tensor_get_size
_LIB.p10_tensor_get_size.argtypes = [P10Tensor]
_LIB.p10_tensor_get_size.restype = c_size_t

######
## p10_tensor_get_dtype
_LIB.p10_tensor_get_dtype.argtypes = [P10Tensor]
_LIB.p10_tensor_get_dtype.restype = DType

######
## p10_tensor_get_shape
_LIB.p10_tensor_get_shape.argtypes = [P10Tensor, POINTER(c_int64), c_size_t]
_LIB.p10_tensor_get_shape.restype = P10ErrorEnum

######
## p10_tensor_get_dimensions
_LIB.p10_tensor_get_dimensions.argtypes = [P10Tensor]
_LIB.p10_tensor_get_dimensions.restype = c_size_t

######
## p10_tensor_get_data
_LIB.p10_tensor_get_data.argtypes = [P10Tensor]
_LIB.p10_tensor_get_data.restype = POINTER(c_void_p)


def _convert_p10_to_ctypes(dtype: DType) -> np.dtype:
    match dtype:
        case DType.FLOAT32:
            return c_float
        case DType.UINT8:
            return c_uint8
        case DType.INT64:
            return c_int64
        case _:
            raise ValueError("Only float32, int64, and uint8 are supported")


def _convert_numpy_to_p10_dtype(dtype: np.dtype) -> DType:
    match dtype:
        case np.float32:
            return DType.FLOAT32
        case np.uint8:
            return DType.UINT8
        case np.int64:
            return DType.INT64
        case _:
            raise ValueError("Only float32, int64, and uint8 are supported")


class Tensor:
    def __init__(self, handle: P10Tensor, data: Optional[np.ndarray] = None):
        if not isinstance(handle, P10Tensor):
            raise ValueError("handle must be a p10_tensor")
        self._handle = handle
        self._data = data

    def __del__(self):
        _LIB.p10_tensor_destroy(self._handle)

    @classmethod
    def from_numpy(cls, array: np.ndarray):
        assert isinstance(array, np.ndarray)
        dtype = _convert_numpy_to_p10_dtype(array.dtype)
        handle = P10Tensor()

        _LIB.p10_tensor_from_data(
            handle,
            dtype,
            array.ctypes.shape_as(c_int64),
            array.ndim,
            array.ctypes.data_as(POINTER(c_uint8)),
        )
        return Tensor(handle, array)

    @contextmanager
    def numpy(self, always_convert=False) -> Generator[np.ndarray, None, None]:
        if self._data is not None and not always_convert:
            yield self._data
            return

        data = _LIB.p10_tensor_get_data(self._handle)

        data = cast(data, POINTER(_convert_p10_to_ctypes(self.dtype())))

        ndim = self.dimensions()
        shape = (c_int64 * ndim)()
        assert (
            _LIB.p10_tensor_get_shape(self._handle, shape, ndim).value
            == P10ErrorEnum.OK
        )

        # Create numpy array from raw pointer
        array = np.ctypeslib.as_array(data, shape=shape).copy()

        yield array

    def shape(self) -> tuple[int, ...]:
        ndim = self.dimensions()
        shape = (c_int64 * ndim)()
        _LIB.p10_tensor_get_shape(self._handle, shape, ndim)
        return tuple(shape)

    def size(self) -> int:
        return _LIB.p10_tensor_get_size(self._handle)

    def dtype(self) -> DType:
        return _LIB.p10_tensor_get_dtype(self._handle)

    def dimensions(self) -> int:
        return _LIB.p10_tensor_get_dimensions(self._handle)

    def c_handle(self) -> P10Tensor:
        """
        Returns the C handle for the tensor.

        **Important**: Keep the tensor alive while the C handle is used, DON'T use this
            Tensor.from_numpy(array).c_handle().

        Returns:
            P10Tensor: The C handle for the tensor.
        """
        return self._handle
