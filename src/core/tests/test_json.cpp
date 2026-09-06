#include <algorithm>
#include <cstddef>
#include <format>
#include <span>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <ptensor/json.hpp>
#include <ptensor/tensor.hpp>
#include <zstd.h>

#include <string_view>

#include "base64_decode.hpp"

namespace p10 {

namespace {
    // Reads a top level field of a flat JSON object, without the surrounding
    // quotes when the value is a string.
    std::string json_field(std::string_view json, std::string_view name);

    std::vector<std::byte> decode_blob(std::string_view json);

    bool same_bytes(std::span<const std::byte> lhs, std::span<const std::byte> rhs);
}  // namespace

TEST_CASE("core::Json describes the tensor layout", "[json]") {
    JsonStaging staging;
    auto tensor = Tensor::from_range(make_shape(2, 3)).expect("Could not create tensor");
    const std::string json = std::format("{}", staging.get_encoder(tensor, JsonEncodeMode::Base64));

    REQUIRE(json.starts_with('{'));
    REQUIRE(json.ends_with('}'));
    REQUIRE(json_field(json, "dtype") == "float32");
    REQUIRE(json_field(json, "shape") == "[2, 3]");
    REQUIRE(json_field(json, "stride") == "[3, 1]");
    REQUIRE(json_field(json, "size_bytes") == "24");
    REQUIRE(json_field(json, "encoding") == "base64");
}

TEST_CASE("core::Json round-trips the tensor bytes", "[json]") {
    const auto mode = GENERATE(JsonEncodeMode::Base64, JsonEncodeMode::Base64Compressed);
    const auto dtype =
        GENERATE(Dtype::Float32, Dtype::Float64, Dtype::Int32, Dtype::Int64, Dtype::Uint8);

    DYNAMIC_SECTION(
        "Encoding " << (mode == JsonEncodeMode::Base64 ? "base64" : "base64+zstd") << " with dtype "
                    << to_string(dtype)
    ) {
        JsonStaging staging;
        auto tensor = Tensor::from_range(make_shape(4, 5), TensorOptions().dtype(dtype))
                          .expect("Could not create tensor");
        const std::string json = std::format("{}", staging.get_encoder(tensor, mode));

        REQUIRE(json_field(json, "dtype") == to_string(dtype));
        REQUIRE(json_field(json, "size_bytes") == std::to_string(tensor.size_bytes()));
        REQUIRE(same_bytes(decode_blob(json), tensor.as_bytes()));
    }
}

TEST_CASE("core::Json compresses by default", "[json]") {
    JsonStaging staging;
    auto tensor = Tensor::full(make_shape(64, 64), 1.0).expect("Could not create tensor");

    const std::string compressed = std::format("{}", staging.get_encoder(tensor));
    const std::string plain =
        std::format("{}", staging.get_encoder(tensor, JsonEncodeMode::Base64));

    REQUIRE(json_field(compressed, "encoding") == "base64+zstd");
    REQUIRE(json_field(plain, "encoding") == "base64");

    // A constant tensor is the easy case for zstd, the blob must shrink a lot.
    REQUIRE(json_field(compressed, "blob").size() < json_field(plain, "blob").size());
    REQUIRE(same_bytes(decode_blob(compressed), tensor.as_bytes()));
}

TEST_CASE("core::Json encodes an empty tensor", "[json]") {
    JsonStaging staging;
    const Tensor tensor;
    const std::string json = std::format("{}", staging.get_encoder(tensor, JsonEncodeMode::Base64));

    REQUIRE(json_field(json, "shape") == "[]");
    REQUIRE(json_field(json, "size_bytes") == "0");
    REQUIRE(json_field(json, "blob").empty());
}

TEST_CASE("core::Json reports the stride of a non-contiguous view", "[json]") {
    JsonStaging staging;
    auto tensor = Tensor::from_range(make_shape(4, 6)).expect("Could not create tensor");
    auto view = tensor.as_slice(Slice(0, 4), Slice(0, 6, 2)).expect("Could not slice tensor");

    const std::string json = std::format("{}", staging.get_encoder(view, JsonEncodeMode::Base64));

    REQUIRE(json_field(json, "shape") == "[4, 3]");
    REQUIRE(json_field(json, "stride") == "[6, 2]");
}

TEST_CASE("core::to_json_debug returns the encoded tensor", "[json]") {
    JsonStaging staging;
    auto tensor = Tensor::from_range(make_shape(2, 3)).expect("Could not create tensor");

    const char* debug_ptr = to_json_debug(tensor);
    REQUIRE(debug_ptr != nullptr);

    const std::string json(debug_ptr);
    REQUIRE(json == std::format("{}", staging.get_encoder(tensor)));
    REQUIRE(json_field(json, "encoding") == "base64+zstd");
    REQUIRE(same_bytes(decode_blob(json), tensor.as_bytes()));
}

TEST_CASE("core::to_json_debug reuses its buffer", "[json]") {
    auto first = Tensor::from_range(make_shape(2, 3)).expect("Could not create tensor");
    auto second = Tensor::from_range(make_shape(4)).expect("Could not create tensor");

    const std::string first_json(to_json_debug(first));
    const std::string second_json(to_json_debug(second));

    REQUIRE(json_field(first_json, "shape") == "[2, 3]");
    REQUIRE(json_field(second_json, "shape") == "[4]");
}

TEST_CASE("core::Json encodes into a wider format string", "[json]") {
    JsonStaging staging;
    auto tensor = Tensor::from_range(make_shape(2)).expect("Could not create tensor");
    const std::string json =
        std::format("[{}]", staging.get_encoder(tensor, JsonEncodeMode::Base64));

    REQUIRE(json.starts_with("[{"));
    REQUIRE(json.ends_with("}]"));
}

namespace {
    // Reads a top level field of a flat JSON object, without the surrounding
    // quotes when the value is a string.
    std::string json_field(std::string_view json, std::string_view name) {
        const std::string key = std::format("\"{}\":", name);
        const size_t key_pos = json.find(key);
        REQUIRE(key_pos != std::string_view::npos);

        size_t start = key_pos + key.size();
        const bool is_string = json[start] == '"';
        if (is_string) {
            start++;
            const size_t end = json.find('"', start);
            REQUIRE(end != std::string_view::npos);
            return std::string(json.substr(start, end - start));
        }

        if (json[start] == '[') {
            const size_t end = json.find(']', start);
            REQUIRE(end != std::string_view::npos);
            return std::string(json.substr(start, end + 1 - start));
        }

        const size_t end = json.find_first_of(",}", start);
        REQUIRE(end != std::string_view::npos);
        return std::string(json.substr(start, end - start));
    }

    std::vector<std::byte> decode_blob(std::string_view json) {
        const auto decoded = tests::base64_decode(json_field(json, "blob"));
        REQUIRE(decoded.has_value());

        if (json_field(json, "encoding") == "base64") {
            return *decoded;
        }

        std::vector<std::byte> plain(std::stoul(json_field(json, "size_bytes")));
        const size_t plain_size =
            ZSTD_decompress(plain.data(), plain.size(), decoded->data(), decoded->size());
        REQUIRE_FALSE(ZSTD_isError(plain_size));
        REQUIRE(plain_size == plain.size());

        return plain;
    }

    bool same_bytes(std::span<const std::byte> lhs, std::span<const std::byte> rhs) {
        return std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
    }
}  // namespace

}  // namespace p10
