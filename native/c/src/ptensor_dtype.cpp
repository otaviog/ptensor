#include "ptensor_dtype.h"

#include <ptensor/dtype.hpp>
#include "dtype_wrapper.hpp"
#include "ptensor_error.h"
#include "update_error_state.hpp"

PTENSOR_API const char* p10_dtype_to_string(P10DTypeEnum dtype) {
    return wrap(dtype).unwrap().to_cstring();
}

PTENSOR_API P10ErrorEnum p10_dtype_from_string(const char* type_str, P10DTypeEnum* out_dtype) {
    auto result = p10::Dtype::from(std::string(type_str));
    if (result.is_error()) {
        p10::update_error_state(result.unwrap_err());
    }
    *out_dtype = static_cast<P10DTypeEnum>(result.unwrap().value);
    return P10ErrorEnum::P10_OK;
}
