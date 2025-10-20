#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <ptensor/testing/catch2_assertions.hpp>

#include "ptensor/p10_error.hpp"
#include "ptensor/shape.hpp"

namespace p10::testing {
TEST_CASE("Testing::Exception", "[testing assertions]") {
    P10Result<int> result = Err(P10Error::InvalidArgument);
    CHECK_THROWS(result.expect("Got error"));
}

TEST_CASE("Testing::IsOk", "[testing assertions]") {
    REQUIRE_THAT(Ok(4), IsOk());
}

TEST_CASE("Testing::compare_tensors", "[testing assertions]") {
    REQUIRE_THAT(
        compare_tensors(
            Tensor::full(make_shape({8, 8}).unwrap(), 123.0).unwrap(),
            Tensor::full(make_shape({8, 8}).unwrap(), 123.0).unwrap()
        ),
        IsOk()
    );

    REQUIRE_THAT(
        compare_tensors(
            Tensor::full(make_shape({8, 7}).unwrap(), 123.0).unwrap(),
            Tensor::full(make_shape({8, 8}).unwrap(), 123.0).unwrap()
        ),
        IsErr()
    );
}
}  // namespace p10::testing
