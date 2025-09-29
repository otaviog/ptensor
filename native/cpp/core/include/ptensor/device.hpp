#pragma once

#include <cinttypes>
#include <string>

#include <ptensor/ptensor_device.h>

#include "ptensor_result.hpp"

namespace p10 {

///
/// Device class to specify the device type and index.
///
struct Device {
    enum Value : uint8_t { Cpu = P10_CPU, Cuda = P10_CUDA, OpenCL = P10_OCL };

    Device() = default;

    /// Constructor
    Device(Value type, int32_t index = -1) : type(type), index(index) {}

    Device(P10DeviceEnum c_enum, int32_t index) : type(static_cast<Value>(c_enum)), index(index) {}

    constexpr operator Value() const {
        return type;
    }

    explicit operator bool() const = delete;

    static PtensorResult<Device> from_int(uint8_t value, int32_t index = -1) {
        if (value > OpenCL) {
            return Err(PtensorError::InvalidArgument);
        }
        return Ok<Device>(Device(static_cast<Value>(value), index));
    }

    std::string to_string() const {
        switch (type) {
            case Cpu:
                return std::string("CPU");
            case Cuda:
                return std::string("CUDA:") + std::to_string(index);
            case OpenCL:
                return std::string("OpenCL:") + std::to_string(index);
            default:
                return "Unknown";
        }
    }

    Value type = Cpu;
    int32_t index = -1;
};

}  // namespace p10
