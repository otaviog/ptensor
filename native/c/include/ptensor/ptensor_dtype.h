#ifndef PTENSOR_DTYPE_H_
#define PTENSOR_DTYPE_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    P10_DTYPE_FLOAT32 = 0,
    P10_DTYPE_FLOAT64,
    P10_DTYPE_FLOAT16,
    P10_DTYPE_UINT8,
    P10_DTYPE_UINT16,
    P10_DTYPE_UINT32,
    P10_DTYPE_INT8,
    P10_DTYPE_INT16,
    P10_DTYPE_INT32,
    P10_DTYPE_INT64,
} P10DTypeEnum;

#define P10_DTYPE_LAST P10_DTYPE_INT64

#ifdef __cplusplus
}
#endif

#endif  // PTENSOR_DTYPE_H_
