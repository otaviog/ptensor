#pragma once

#if defined(_MSC_VER)
    #define PTENSOR_AVX2
#elif defined(__x86_64__) || defined(__i386__)
    #define PTENSOR_AVX2 __attribute__((target("avx2")))
#else
    #define PTENSOR_AVX2
#endif
