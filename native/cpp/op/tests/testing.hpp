#pragma once

#include <filesystem>

#include <ptensor/tensor.hpp>

namespace p10::testing {
std::filesystem::path get_output_path();

namespace samples {
    Tensor image01();
}
}  // namespace p10::testing