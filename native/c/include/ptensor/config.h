#ifndef PTENSOR_CONFIG_H
#define PTENSOR_CONFIG_H

#include <stddef.h>

#ifdef _WIN32
    #if defined(PTENSOR_BUILD_SHARED) /* build dll */
        #define PTENSOR_API __declspec(dllexport)
    #elif defined(PTENSOR_BUILD_STATIC) /* static library */
        #define PTENSOR_API
    #else /* use dll */ 
        #define PTENSOR_API __declspec(dllimport)
    #endif
#elif __GNUC__ >= 4
    #if defined(PTENSOR_BUILD_SHARED) /* build dll */
        #define PTENSOR_API __attribute__((visibility("default")))
    #else
        #define PTENSOR_API
    #endif
#else
    #define PTENSOR_API
#endif

const int PTENSOR_VERSION_MAJOR = 1;
const int PTENSOR_VERSION_MINOR = 0;
const int PTENSOR_VERSION = ((PTENSOR_VERSION_MAJOR << 16) | PTENSOR_VERSION_MINOR);
const unsigned long P10_MAX_SHAPE = 8;

#ifdef __cplusplus
#if defined __has_include && __has_include(<Windows.h>)
#define PTENSOR_HAS_WINDOWS_H
#endif
#endif

#ifdef _MSC_VER
    // Microsoft Visual C++
    #define PTENSOR_PACKED_BEGIN __pragma(pack(push, 1))
    #define PTENSOR_PACKED_END __pragma(pack(pop))
    #define PTENSOR_PACKED_STRUCT struct
#elif defined(__GNUC__) || defined(__clang__)
    // GCC, Clang, and compatible compilers
    #define PTENSOR_PACKED_BEGIN
    #define PTENSOR_PACKED_END
    #define PTENSOR_PACKED_STRUCT struct __attribute__((packed))
#else
    // Fallback for other compilers
    #define PTENSOR_PACKED_BEGIN
    #define PTENSOR_PACKED_END
    #define PTENSOR_PACKED_STRUCT struct
    #warning "Packed structs may not work correctly on this compiler"
#endif

// Helper macro for creating packed structs
#define PTENSOR_DEFINE_PACKED_STRUCT(name) \
    PTENSOR_PACKED_BEGIN \
    PTENSOR_PACKED_STRUCT name

#endif  // PTENSOR_CONFIG_H
