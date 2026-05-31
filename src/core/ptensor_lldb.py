"""LLDB summary providers for ptensor types.

Mirrors `src/core/ptensor.natvis` for debuggers that don't read natvis
(lldb-dap, plain lldb). Stats (min/max/mean) come from the
`p10::tensor_*_debug` helpers in `ptensor/tensor_print.hpp`, invoked through
LLDB's expression evaluator.

Load from the LLDB CLI:

    command script import path/to/ptensor_lldb.py

Or via VS Code launch.json (lldb-dap):

    "initCommands": [
        "command script import ${workspaceFolder}/src/core/ptensor_lldb.py"
    ]
"""

import lldb

DTYPE_NAMES = {
    0: "float32",
    1: "float64",
    2: "float16",
    3: "uint8",
    4: "uint16",
    5: "uint32",
    6: "int8",
    7: "int16",
    8: "int32",
    9: "int64",
}


def _extents_to_list(value):
    """Read a `p10::detail::TensorExtents`-shaped SBValue as a list[int]."""
    dims_v = value.GetChildMemberWithName("dims_")
    extent = value.GetChildMemberWithName("extent_")
    if not dims_v.IsValid() or not extent.IsValid():
        return None
    dims = dims_v.GetValueAsUnsigned()
    data = extent.GetData()
    err = lldb.SBError()
    out = []
    for i in range(dims):
        v = data.GetSignedInt64(err, i * 8)
        if err.Fail():
            return None
        out.append(v)
    return out


def _format_extents(value):
    items = _extents_to_list(value)
    if items is None:
        return "<unreadable>"
    return "[" + ", ".join(str(d) for d in items) + "]"


def dtype_summary(value, internal_dict):
    code_v = value.GetChildMemberWithName("value")
    if not code_v.IsValid():
        return "<unreadable>"
    code = code_v.GetValueAsUnsigned()
    return DTYPE_NAMES.get(code, f"dtype({code})")


def extents_summary(value, internal_dict):
    return _format_extents(value)


def _call_double(target, expr):
    options = lldb.SBExpressionOptions()
    options.SetIgnoreBreakpoints(True)
    options.SetTryAllThreads(False)
    options.SetTimeoutInMicroSeconds(2_000_000)
    result = target.EvaluateExpression(expr, options)
    if result.GetError().Fail():
        return None
    text = result.GetValue()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def tensor_summary(value, internal_dict):
    shape = value.GetChildMemberWithName("shape_")
    dtype = value.GetChildMemberWithName("dtype_")
    parts = [
        f"shape={_format_extents(shape)}",
        f"dtype={dtype_summary(dtype, internal_dict)}",
    ]

    addr = value.GetLoadAddress()
    if addr != lldb.LLDB_INVALID_ADDRESS and addr != 0:
        target = value.GetTarget()
        cast = f"*(const p10::Tensor*)0x{addr:x}"
        for label, fn in (
            ("min", "tensor_min_debug"),
            ("max", "tensor_max_debug"),
            ("mean", "tensor_mean_debug"),
        ):
            result = _call_double(target, f"p10::{fn}({cast})")
            if result is not None:
                parts.append(f"{label}={result:g}")

    return "Tensor(" + ", ".join(parts) + ")"


def __lldb_init_module(debugger, internal_dict):
    mod = __name__
    commands = (
        f"type summary add -F {mod}.tensor_summary p10::Tensor",
        f"type summary add -F {mod}.extents_summary p10::Shape",
        f"type summary add -F {mod}.extents_summary p10::Stride",
        f"type summary add -F {mod}.extents_summary p10::detail::TensorExtents",
        f"type summary add -F {mod}.dtype_summary p10::Dtype",
    )
    for cmd in commands:
        debugger.HandleCommand(cmd)
