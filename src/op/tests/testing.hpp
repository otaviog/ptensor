#pragma once

#include <tuple>

#include <ptensor/tensor.hpp>
#include <ptensor/testing/output_path.hpp>  // get_output_path, suffixed

namespace p10::testing::samples {

std::tuple<Tensor, std::string> image01();

}  // namespace p10::testing::samples
