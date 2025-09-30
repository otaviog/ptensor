#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <ptensor/testing/assertions.hpp>

#include "ptensor/ptensor_error.hpp"
#include "ptensor/shape.hpp"

namespace p10::testing {
TEST_CASE("Testing::Exception", "[testing assertions]") {
    PtensorResult<int> result = Err<int>(PtensorError::InvalidArgument);
    CHECK_THROWS(result.expect("Got error"));
}

TEST_CASE("Testing::IsOk", "[testing assertions]") {
    REQUIRE_THAT(Ok<int>(4), IsOk());
}

TEST_CASE("Testing::compare_tensors", "[testing assertions]") {
    REQUIRE_THAT(
        compare_tensors(
            Tensor::full(make_shape({8, 8}).unwrap(), 123.0),
            Tensor::full(make_shape({8, 8}).unwrap(), 123.0)
        ),
        IsOk()
    );

    REQUIRE_THAT(
        compare_tensors(
            Tensor::full(make_shape({8, 7}).unwrap(), 123.0),
            Tensor::full(make_shape({8, 8}).unwrap(), 123.0)
        ),
        IsErr()
    );
}
}  // namespace p10::testing
