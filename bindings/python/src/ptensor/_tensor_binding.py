import ctypes
from ctypes import (
    c_int,
    c_void_p,
    POINTER,
    c_size_t,
    c_int64,
    c_uint8,
    c_uint16,
    c_uint32,
    c_float,
    c_double,
    c_int8,
    c_int16,
    c_int32,
    cast,
    byref,
)
from enum import IntEnum

import numpy as np

from ._core_binding import get_library, P10ErrorEnum, P10Error

P10Tensor = c_void_p

_LIB = get_library()


class DType(IntEnum):
    FLOAT32 = 0
    FLOAT64 = 1
    FLOAT16 = 2
    UINT8 = 3
    UINT16 = 4
    UINT32 = 5
    INT8 = 6
    INT16 = 7
    INT32 = 8
    INT64 = 9


# p10_dtype_to_string
_LIB.p10_dtype_to_string.argtypes = [c_int]
_LIB.p10_dtype_to_string.restype = ctypes.c_char_p

# p10_dtype_from_string
_LIB.p10_dtype_from_string.argtypes = [ctypes.c_char_p, POINTER(c_int)]
_LIB.p10_dtype_from_string.restype = c_int

# p10_dtype_size_bytes
_LIB.p10_dtype_size_bytes.argtypes = [c_int]
_LIB.p10_dtype_size_bytes.restype = c_size_t

# p10_from_data
_LIB.p10_from_data.argtypes = [POINTER(P10Tensor), c_int, POINTER(c_int64), c_size_t, c_void_p]
_LIB.p10_from_data.restype = c_int

# p10_from_data_strided
_LIB.p10_from_data_strided.argtypes = [
    POINTER(P10Tensor),
    c_int,
    POINTER(c_int64),
    POINTER(c_int64),
    c_size_t,
    c_void_p,
]
_LIB.p10_from_data_strided.restype = c_int

# p10_destroy
_LIB.p10_destroy.argtypes = [POINTER(P10Tensor)]
_LIB.p10_destroy.restype = c_int

# p10_get_size
_LIB.p10_get_size.argtypes = [P10Tensor]
_LIB.p10_get_size.restype = c_size_t

# p10_get_size_bytes
_LIB.p10_get_size_bytes.argtypes = [P10Tensor]
_LIB.p10_get_size_bytes.restype = c_size_t

# p10_get_dtype
_LIB.p10_get_dtype.argtypes = [P10Tensor]
_LIB.p10_get_dtype.restype = c_int

# p10_get_shape
_LIB.p10_get_shape.argtypes = [P10Tensor, POINTER(c_int64), c_size_t]
_LIB.p10_get_shape.restype = c_int

# p10_get_stride
_LIB.p10_get_stride.argtypes = [P10Tensor, POINTER(c_int64), c_size_t]
_LIB.p10_get_stride.restype = c_int

# p10_get_ndim
_LIB.p10_get_ndim.argtypes = [P10Tensor]
_LIB.p10_get_ndim.restype = c_size_t

# p10_get_data
_LIB.p10_get_data.argtypes = [P10Tensor]
_LIB.p10_get_data.restype = c_void_p

# p10_is_empty
_LIB.p10_is_empty.argtypes = [P10Tensor]
_LIB.p10_is_empty.restype = c_int


_DTYPE_TO_NUMPY: dict[DType, np.dtype] = {
    DType.FLOAT32: np.dtype("float32"),
    DType.FLOAT64: np.dtype("float64"),
    DType.FLOAT16: np.dtype("float16"),
    DType.UINT8: np.dtype("uint8"),
    DType.UINT16: np.dtype("uint16"),
    DType.UINT32: np.dtype("uint32"),
    DType.INT8: np.dtype("int8"),
    DType.INT16: np.dtype("int16"),
    DType.INT32: np.dtype("int32"),
    DType.INT64: np.dtype("int64"),
}

_NUMPY_TO_DTYPE: dict[np.dtype, DType] = {v: k for k, v in _DTYPE_TO_NUMPY.items()}

_DTYPE_TO_CTYPE = {
    DType.FLOAT32: c_float,
    DType.FLOAT64: c_double,
    DType.FLOAT16: c_uint16,  # no float16 in ctypes; same storage size
    DType.UINT8: c_uint8,
    DType.UINT16: c_uint16,
    DType.UINT32: c_uint32,
    DType.INT8: c_int8,
    DType.INT16: c_int16,
    DType.INT32: c_int32,
    DType.INT64: c_int64,
}


def dtype_to_numpy(dtype: DType) -> np.dtype:
    """Converts a ptensor DType to its corresponding numpy dtype."""
    try:
        return _DTYPE_TO_NUMPY[dtype]
    except KeyError:
        raise ValueError(f"Unsupported DType for numpy conversion: {dtype}")


def dtype_from_numpy(dtype: np.dtype) -> DType:
    """Converts a numpy dtype to its corresponding ptensor DType."""
    dt = np.dtype(dtype)
    try:
        return _NUMPY_TO_DTYPE[dt]
    except KeyError:
        raise ValueError(f"Unsupported numpy dtype: {dtype}")


def _check(err: int) -> None:
    """Raises P10Error if err is not OK."""
    code = P10ErrorEnum(err)
    if code != P10ErrorEnum.OK:
        raise P10Error(code)


class Tensor:
    """A ptensor Tensor wrapping the C API handle.

    The tensor may be backed by a numpy array (keeping it alive) or own its
    data through the C library.
    """

    def __init__(self, handle: int, owner: object = None):
        """
        Args:
            handle: Raw integer value of a P10Tensor (c_void_p).
            owner:  Optional object to keep alive while this Tensor exists
                    (e.g. the numpy array that owns the buffer).
        """
        if handle is None or handle == 0:
            raise ValueError("Tensor handle must be non-null")
        self._handle = handle
        self._owner = owner

    def __del__(self):
        if self._handle:
            h = P10Tensor(self._handle)
            _LIB.p10_destroy(byref(h))
            self._handle = 0

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> "Tensor":
        """Creates a Tensor that is a view over *array*'s data.

        The array is kept alive for the lifetime of the Tensor.
        Non-contiguous arrays (e.g. transposed views) are handled via strides.
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("array must be a numpy ndarray")

        dtype = dtype_from_numpy(array.dtype)
        shape = (c_int64 * array.ndim)(*array.shape)

        elem_size = array.itemsize
        # numpy strides are in bytes; ptensor strides are in elements
        py_strides = tuple(s // elem_size for s in array.strides)

        data_ptr = array.ctypes.data_as(c_void_p)
        handle = P10Tensor()

        if array.flags["C_CONTIGUOUS"]:
            _check(
                _LIB.p10_from_data(byref(handle), c_int(dtype), shape, array.ndim, data_ptr)
            )
        else:
            c_strides = (c_int64 * array.ndim)(*py_strides)
            _check(
                _LIB.p10_from_data_strided(
                    byref(handle),
                    c_int(dtype),
                    shape,
                    c_strides,
                    array.ndim,
                    data_ptr,
                )
            )

        return cls(handle.value, array)

    # ------------------------------------------------------------------ #
    # Data access
    # ------------------------------------------------------------------ #

    def numpy(self) -> np.ndarray:
        """Returns a numpy array sharing the tensor's data buffer.

        The returned array is valid only as long as this Tensor is alive.
        For owned tensors (not backed by a numpy array), the data is copied.
        """
        if self._owner is not None and isinstance(self._owner, np.ndarray):
            return self._owner

        ctype = _DTYPE_TO_CTYPE[self.dtype]
        data_ptr = _LIB.p10_get_data(self._handle)
        data = cast(data_ptr, POINTER(ctype))

        shape = self.shape
        ndim = len(shape)
        strides_elems = self.strides
        strides_bytes = tuple(s * self.dtype_size_bytes for s in strides_elems)

        # Build a numpy array that points at the C buffer (no copy).
        arr = np.ctypeslib.as_array(data, shape=shape)
        if strides_bytes != tuple(arr.strides):
            arr = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides_bytes)
        return arr.copy()

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def dtype(self) -> DType:
        return DType(_LIB.p10_get_dtype(self._handle))

    @property
    def dtype_size_bytes(self) -> int:
        return _LIB.p10_dtype_size_bytes(c_int(self.dtype))

    @property
    def ndim(self) -> int:
        return _LIB.p10_get_ndim(self._handle)

    @property
    def shape(self) -> tuple[int, ...]:
        n = self.ndim
        buf = (c_int64 * n)()
        _check(_LIB.p10_get_shape(self._handle, buf, n))
        return tuple(int(x) for x in buf)

    @property
    def strides(self) -> tuple[int, ...]:
        """Per-element strides (not byte strides)."""
        n = self.ndim
        buf = (c_int64 * n)()
        _check(_LIB.p10_get_stride(self._handle, buf, n))
        return tuple(int(x) for x in buf)

    @property
    def size(self) -> int:
        """Total number of elements."""
        return _LIB.p10_get_size(self._handle)

    @property
    def size_bytes(self) -> int:
        """Total size in bytes."""
        return _LIB.p10_get_size_bytes(self._handle)

    @property
    def is_empty(self) -> bool:
        return bool(_LIB.p10_is_empty(self._handle))

    def c_handle(self) -> int:
        """Returns the raw integer value of the C handle.

        Keep this Tensor alive while the handle is in use.
        """
        return self._handle

    def __repr__(self) -> str:
        if self.is_empty:
            return "Tensor(empty)"
        return f"Tensor(shape={self.shape}, dtype={self.dtype.name})"
