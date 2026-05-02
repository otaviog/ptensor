#ifndef PTENSOR_DEVICE_H
#define PTENSOR_DEVICE_H

typedef enum {
    P10_CPU = 0,
    P10_CUDA,
    P10_OCL
} P10DeviceEnum;

#endif  // PTENSOR_DEVICE_H
