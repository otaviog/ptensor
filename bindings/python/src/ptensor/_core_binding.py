from pathlib import Path
import ctypes
from enum import IntEnum


_LIB = None

_SEARCH_PATHS = None


def _build_search_paths():
    root = Path(__file__).absolute().parent.parent.parent.parent.parent

    # cmake workflow preset output locations — build tree first (rpath is intact),
    # then install tree
    return [
        root / "build/clang/debug/src/c",
        root / "build/clang/release/src/c",
        root / "build/clang/installed/debug/lib",
        root / "build/clang/installed/release/lib",
        root / "build/msbuild/install/lib",
        root / "build/msbuild/install/bin",
    ]


def get_library() -> ctypes.CDLL:
    global _LIB

    if _LIB is not None:
        return _LIB

    _SEARCH_FILES = ["libptensor_capi.dylib", "libptensor_capi.so", "ptensor_capi.dll"]
    for path in _build_search_paths():
        if not path.exists():
            continue
        for filename in _SEARCH_FILES:
            dll_path = (path / filename).resolve()
            if dll_path.exists():
                _LIB = ctypes.CDLL(str(dll_path))
                return _LIB

    raise OSError(
        "ptensor_capi shared library not found. "
        "Build the project first with: cmake --workflow --preset clang/debug/install"
    )


class P10ErrorEnum(IntEnum):
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
    INFER_ERROR = 10


get_library()

# p10_get_last_error_message
_LIB.p10_get_last_error_message.restype = ctypes.c_char_p
_LIB.p10_get_last_error_message.argtypes = []


def get_last_error_message() -> str:
    msg = _LIB.p10_get_last_error_message()
    return msg.decode("utf-8") if msg else ""


class P10Error(Exception):
    def __init__(self, error_code: P10ErrorEnum):
        self.error_code = P10ErrorEnum(error_code)
        self.error_message = get_last_error_message() if self.error_code != P10ErrorEnum.OK else ""

    def __str__(self):
        return f"P10Error({self.error_code.name}): {self.error_message}"

    def raise_if_error(self):
        if self.error_code != P10ErrorEnum.OK:
            raise self
