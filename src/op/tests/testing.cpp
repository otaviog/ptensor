#include "testing.hpp"

#include <ptensor/io/image.hpp>

namespace p10::testing::samples {
    std::tuple<Tensor, std::string> image01() {
        const std::string image = "image01.png";
        return {io::load_image("tests/data/image/" + image).expect("Can't load test image"), image};
    }
}  // namespace p10::testing::samples
