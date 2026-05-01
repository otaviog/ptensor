#include <cstdint>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <ptensor/dtype.hpp>

namespace p10 {

TEST_CASE("Dtype::construction and enum values", "[dtype]") {
    SECTION("Default construction") {
        Dtype dtype;
        REQUIRE(dtype == Dtype::Float32);
    }

    SECTION("Value construction") {
        REQUIRE(Dtype(Dtype::Float32) == Dtype::Float32);
        REQUIRE(Dtype(Dtype::Float64) == Dtype::Float64);
        REQUIRE(Dtype(Dtype::Float16) == Dtype::Float16);
        REQUIRE(Dtype(Dtype::Uint8) == Dtype::Uint8);
        REQUIRE(Dtype(Dtype::Uint16) == Dtype::Uint16);
        REQUIRE(Dtype(Dtype::Uint32) == Dtype::Uint32);
        REQUIRE(Dtype(Dtype::Int8) == Dtype::Int8);
        REQUIRE(Dtype(Dtype::Int16) == Dtype::Int16);
        REQUIRE(Dtype(Dtype::Int32) == Dtype::Int32);
        REQUIRE(Dtype(Dtype::Int64) == Dtype::Int64);
    }

    SECTION("Implicit conversion to Code") {
        Dtype dtype(Dtype::Float64);
        Dtype::Code value = dtype;
        REQUIRE(value == Dtype::Float64);
    }
}

TEST_CASE("Dtype::from() template function", "[dtype]") {
    SECTION("Float types") {
        REQUIRE(Dtype::from<float>() == Dtype::Float32);
        REQUIRE(Dtype::from<double>() == Dtype::Float64);
    }

    SECTION("Unsigned integer types") {
        REQUIRE(Dtype::from<uint8_t>() == Dtype::Uint8);
        REQUIRE(Dtype::from<unsigned char>() == Dtype::Uint8);
        REQUIRE(Dtype::from<uint16_t>() == Dtype::Uint16);
        REQUIRE(Dtype::from<uint32_t>() == Dtype::Uint32);
    }

    SECTION("Signed integer types") {
        REQUIRE(Dtype::from<int8_t>() == Dtype::Int8);
        REQUIRE(Dtype::from<int16_t>() == Dtype::Int16);
        REQUIRE(Dtype::from<int32_t>() == Dtype::Int32);
        REQUIRE(Dtype::from<int64_t>() == Dtype::Int64);
    }

    SECTION("CV-qualified types") {
        REQUIRE(Dtype::from<const float>() == Dtype::Float32);
        REQUIRE(Dtype::from<volatile double>() == Dtype::Float64);
        REQUIRE(Dtype::from<const volatile int32_t>() == Dtype::Int32);
        REQUIRE(Dtype::from<const volatile uint8_t>() == Dtype::Uint8);
    }

    SECTION("Reference types") {
        REQUIRE(Dtype::from<float&>() == Dtype::Float32);
        REQUIRE(Dtype::from<const double&>() == Dtype::Float64);
        REQUIRE(Dtype::from<int32_t&&>() == Dtype::Int32);
    }
}

TEST_CASE("Dtype::size() method", "[dtype]") {
    SECTION("1-byte types") {
        REQUIRE(Dtype(Dtype::Uint8).size_bytes() == 1);
        REQUIRE(Dtype(Dtype::Int8).size_bytes() == 1);
    }

    SECTION("2-byte types") {
        REQUIRE(Dtype(Dtype::Float16).size_bytes() == 2);
        REQUIRE(Dtype(Dtype::Uint16).size_bytes() == 2);
        REQUIRE(Dtype(Dtype::Int16).size_bytes() == 2);
    }

    SECTION("4-byte types") {
        REQUIRE(Dtype(Dtype::Uint32).size_bytes() == 4);
        REQUIRE(Dtype(Dtype::Int32).size_bytes() == 4);
        REQUIRE(Dtype(Dtype::Float32).size_bytes() == 4);
    }

    SECTION("8-byte types") {
        REQUIRE(Dtype(Dtype::Int64).size_bytes() == 8);
        REQUIRE(Dtype(Dtype::Float64).size_bytes() == 8);
    }
}

TEST_CASE("Dtype::visit() functionality", "[dtype]") {
    SECTION("Visit with mutable data") {
        std::vector<std::byte> data(16);

        auto visitor = [](auto typed_span) {
            using T = typename decltype(typed_span)::element_type;
            for (size_t i = 0; i < typed_span.size(); ++i) {
                typed_span[i] = static_cast<T>(i + 1);
            }
        };

        Dtype float32_dtype(Dtype::Float32);
        float32_dtype.visit(visitor, std::span(data));

        auto float_span =
            std::span(reinterpret_cast<float*>(data.data()), data.size() / sizeof(float));
        REQUIRE(float_span[0] == 1.0f);
        REQUIRE(float_span[1] == 2.0f);
        REQUIRE(float_span[2] == 3.0f);
        REQUIRE(float_span[3] == 4.0f);
    }

    SECTION("Visit with const data") {
        std::vector<uint32_t> source_data = {10, 20, 30, 40};
        std::vector<std::byte> data(source_data.size() * sizeof(uint32_t));
        std::memcpy(data.data(), source_data.data(), data.size());

        uint64_t sum = 0;
        auto visitor = [&sum](auto typed_span) {
            for (const auto& val : typed_span) {
                sum += uint64_t(val);
            }
        };

        Dtype uint32_dtype(Dtype::Uint32);
        uint32_dtype.visit(visitor, std::span<const std::byte>(data));
        REQUIRE(sum == 100);
    }

    SECTION("Visit with different data types") {
        std::vector<std::byte> data(8);

        auto counting_visitor = [](auto typed_span) {
            using T = typename decltype(typed_span)::element_type;
            for (size_t i = 0; i < typed_span.size(); ++i) {
                typed_span[i] = static_cast<T>(i);
            }
        };

        SECTION("Int8") {
            Dtype dtype(Dtype::Int8);
            dtype.visit(counting_visitor, std::span(data));
            auto int8_span = std::span(reinterpret_cast<int8_t*>(data.data()), 8);
            REQUIRE(int8_span[0] == 0);
            REQUIRE(int8_span[7] == 7);
        }

        SECTION("Uint16") {
            Dtype dtype(Dtype::Uint16);
            dtype.visit(counting_visitor, std::span(data));
            auto uint16_span = std::span(reinterpret_cast<uint16_t*>(data.data()), 4);
            REQUIRE(uint16_span[0] == 0);
            REQUIRE(uint16_span[3] == 3);
        }

        SECTION("Float64") {
            Dtype dtype(Dtype::Float64);
            dtype.visit(counting_visitor, std::span(data));
            auto double_span = std::span(reinterpret_cast<double*>(data.data()), 1);
            REQUIRE(double_span[0] == 0.0);
        }
    }
}

TEST_CASE("Dtype::match() functionality", "[dtype]") {
    SECTION("Integer matcher") {
        auto int_matcher = [](auto type_id) {
            using T = typename decltype(type_id)::type;
            return sizeof(T);
        };

        auto float_matcher = [](auto) {
            return -1;  // Should not be called for integer types
        };

        REQUIRE(Dtype::from<int8_t>().match(int_matcher, float_matcher) == 1);
        REQUIRE(Dtype::from<uint16_t>().match(int_matcher, float_matcher) == 2);
        REQUIRE(Dtype::from<int32_t>().match(int_matcher, float_matcher) == 4);
        REQUIRE(Dtype::from<int64_t>().match(int_matcher, float_matcher) == 8);
    }

    SECTION("Float matcher") {
        auto int_matcher = [](auto) {
            return -1;  // Should not be called for float types
        };

        auto float_matcher = [](auto type_id) {
            using T = typename decltype(type_id)::type;
            return sizeof(T);
        };

        REQUIRE(Dtype::from<float>().match(int_matcher, float_matcher) == 4);
        REQUIRE(Dtype::from<double>().match(int_matcher, float_matcher) == 8);
    }
}

TEST_CASE("Dtype::to_string() function", "[dtype]") {
    REQUIRE(to_string(Dtype::Float32).length() > 0);
    REQUIRE(to_string(Dtype::Float64).length() > 0);
    REQUIRE(to_string(Dtype::Float16).length() > 0);
    REQUIRE(to_string(Dtype::Uint8).length() > 0);
    REQUIRE(to_string(Dtype::Uint16).length() > 0);
    REQUIRE(to_string(Dtype::Uint32).length() > 0);
    REQUIRE(to_string(Dtype::Int8).length() > 0);
    REQUIRE(to_string(Dtype::Int16).length() > 0);
    REQUIRE(to_string(Dtype::Int32).length() > 0);
    REQUIRE(to_string(Dtype::Int64).length() > 0);
}

TEST_CASE("Dtype::edge cases and error conditions", "[dtype]") {
    SECTION("Visit with invalid dtype") {
        std::vector<std::byte> data(8);
        auto visitor = [](auto) {};

        Dtype invalid_dtype;
        invalid_dtype.value = static_cast<Dtype::Code>(255);

        REQUIRE_THROWS_AS(invalid_dtype.visit(visitor, std::span(data)), std::runtime_error);
    }

    SECTION("Size of invalid dtype") {
        Dtype invalid_dtype;
        invalid_dtype.value = static_cast<Dtype::Code>(255);
        REQUIRE(invalid_dtype.size_bytes() == 0);
    }
}

}  // namespace p10