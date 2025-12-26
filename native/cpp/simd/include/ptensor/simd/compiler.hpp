#pragma once

#ifdef _MSC_VER
    #define PTENSOR_AVX2
#else
    #define PTENSOR_AVX2 __attribute__((target("avx2")))
#endif
