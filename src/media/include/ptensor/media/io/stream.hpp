#pragma once

#include <vector>

#include "audio_parameters.hpp"
#include "video_parameters.hpp"

namespace p10::media {

struct Device {    
    std::string name;
    int device_index;
    std::optional<VideoParameters> video_params;
    std::optional<AudioParameters> audio_params;
};

std::vector<Device> listMediaDevices();

}
