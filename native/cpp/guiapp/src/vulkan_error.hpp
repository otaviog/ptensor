#pragma once
#include <string>

#include <ptensor/p10_error.hpp>
#include <vulkan/vulkan.h>

namespace p10::guiapp {

P10Error wrap_vk_result(VkResult err);
}