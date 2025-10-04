#pragma once

#include <ptensor/dtype.hpp>

#include "ptensor_dtype.h"

namespace p10 {
inline PtensorResult<Dtype> wrap(P10DTypeEnum dtype) {
    if (dtype < 0 || dtype > P10_DTYPE_LAST) {
        return Err(PtensorError::InvalidArgument, "Invalid dtype");
    }
    return Ok(Dtype(static_cast<Dtype::Code>(dtype)));
}
}  // namespace p10

using p10::wrap;
