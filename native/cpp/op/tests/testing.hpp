#pragma once

#include <tuple>
#include <filesystem>

#include <ptensor/tensor.hpp>

namespace p10::testing {

std::string suffixed(const std::string& filename, const std::string& suffix);

std::filesystem::path get_output_path();

namespace samples {
    std::tuple<Tensor, std::string> image01();
}
}  // namespace p10::testing