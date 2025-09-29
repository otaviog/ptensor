#ifndef PTENSOR_ERROR_H
#define PTENSOR_ERROR_H

#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    P10_OK = 0,
    P10_UNKNOWN_ERROR,
    P10_ASSERTION_ERROR,
    P10_INVALID_ARGUMENT,
    P10_INVALID_OPERATION,
    P10_OUT_OF_MEMORY,
    P10_OUT_OF_RANGE, 
    P10_NOT_IMPLEMENTED, 
    P10_OS_ERROR,
    P10_IO_ERROR
} P10ErrorEnum;

PTENSOR_API const char* p10_get_last_error_message();

#ifdef __cplusplus
}
#endif

#endif
