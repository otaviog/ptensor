#pragma once
#include <string>

#include <ptensor/p10_error.hpp>
#include <vulkan/vulkan.h>

namespace p10::viz {

P10Error wrap_vk_result(VkResult err);
}