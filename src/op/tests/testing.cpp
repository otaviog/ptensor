#include "testing.hpp"

#include <ptensor/io/image.hpp>

namespace p10::testing {

std::string suffixed(const std::string& filename, const std::string& suffix) {
    std::filesystem::path path(filename);
    std::string stem = path.stem().string();
    return stem + "-" + suffix + path.extension().string();
}

std::filesystem::path get_output_path() {
    std::filesystem::path output_path("native/cpp/tests/output");
    std::filesystem::create_directories(output_path);
    return output_path;
}

namespace samples {
    std::tuple<Tensor, std::string> image01() {
        const std::string image = "image01.png";
        return {io::load_image("tests/data/image/" + image).expect("Can't load test image"), image};
    }
}  // namespace samples
}  // namespace p10::testing
