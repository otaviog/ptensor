#pragma once

#include <vector>

#include "audio_parameters.hpp"
#include "video_parameters.hpp"

namespace p10::media {

/// Media device information.
struct Device {
    /// Human-readable device name.
    std::string name;
    /// Device index.
    int device_index;
    /// Video parameters if available.
    std::optional<VideoParameters> video_params;
    /// Audio parameters if available.
    std::optional<AudioParameters> audio_params;
};

/// List available media devices.
std::vector<Device> listMediaDevices();

}  // namespace p10::media
