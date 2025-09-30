from pathlib import Path
import ctypes
from enum import IntEnum


_LIB = None


def get_library() -> ctypes.CDLL:
    global _LIB

    if _LIB is not None:
        return _LIB
    # Load the shared library

    _current_dir = Path(__file__).absolute().parent
    _SEARCH_PATHS = [
        _current_dir / "../../build/lin-release/native/c",
        _current_dir / "../../build/lin-debug/native/c",
        _current_dir / "../../build/msbuild/native/c/Release",
        _current_dir / "../../build/msbuild/native/c/Debug",
    ]
    _SEARCH_FILES = ["libptensor_capi.so", "ptensor_capi.dll"]
    for path in _SEARCH_PATHS:
        if not path.exists():
            continue
        path = path.resolve()
        try:
            dll_file = next(
                filter(
                    lambda dll_file: dll_file.exists(),
                    map(lambda x, path=path: (path / x), _SEARCH_FILES),
                )
            )
        except StopIteration:
            raise OSError("Failed to find the shared library")
        _LIB = ctypes.CDLL(str(dll_file))
        break
    return _LIB


class PtensorErrorEnum(IntEnum):
    OK = 0
    UNKNOWN_ERROR = 1
    ASSERTION_ERROR = 2
    INVALID_ARGUMENT = 3
    INVALID_OPERATION = 4
    OUT_OF_MEMORY = 5
    OUT_OF_RANGE = 6
    NOT_IMPLEMENTED = 7
    OS_ERROR = 8
    IO_ERROR = 9


get_library()

# p10_get_last_error_message
_LIB.p10_get_last_error_message.restype = ctypes.c_char_p


def get_last_error_message():
    return _LIB.p10_get_last_error_message().decode("utf-8")


class PtensorError(Exception):
    def __init__(self, error_code: PtensorErrorEnum):
        self.error_code = error_code
        if error_code != PtensorErrorEnum.OK:
            self.error_message = get_last_error_message()

    def __str__(self):
        return f"PtensorError: {self.error_message}"

    def raise_if_error(self):
        if self.error_code != PtensorErrorEnum.OK:
            raise self
