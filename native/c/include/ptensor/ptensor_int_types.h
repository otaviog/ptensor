#ifndef P10_TYPES_H
#define P10_TYPES_H

typedef char P10Int8;
typedef unsigned char P10Uint8;
typedef short P10Int16;
typedef unsigned short P10Uint16;
typedef int P10Int32;
typedef unsigned int P10Uint32;
#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__)
    #if defined(_WIN64)
typedef long long int P10Int64;
typedef unsigned long long int P10Uint64;
    #else
typedef long int P10Int64;
typedef unsigned long int P10Uint64;
    #endif
#else
typedef long long int P10Int64;
typedef unsigned long long int P10Uint64;
#endif
typedef P10Uint64 P10Size;

#endif