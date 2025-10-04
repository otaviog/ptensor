#include <catch2/catch_test_macros.hpp>
#include <ptensor/ptensor_error.hpp>
#include <ptensor/ptensor_result.hpp>

namespace p10 {

TEST_CASE("PtensorError::(to_string() and code())", "[error]") {
    PtensorError error = PtensorError::InvalidOperation;
    REQUIRE(error.to_string() == "Invalid operation");
    REQUIRE(error.code() == PtensorError::InvalidOperation);

    error = PtensorError::InvalidArgument;
    REQUIRE(error.to_string() == "Invalid argument");
    REQUIRE(error.code() == PtensorError::InvalidArgument);

    error = PtensorError::NotImplemented;
    REQUIRE(error.to_string() == "Not implemented");
    REQUIRE(error.code() == PtensorError::NotImplemented);
}

TEST_CASE("PtensorError::fromAssert()", "[error]") {
    PtensorError error = PtensorError::fromAssert("assertion failed", "file.cpp", 42);
    REQUIRE(error.code() == PtensorError::AssertionError);
    REQUIRE(error.to_string() == "Assertion error: assertion failed (file.cpp:42)");
}

TEST_CASE("PtensorResult::(Ok and Err)", "[result]") {
    PtensorResult<int> ok = Ok<int>(42);
    REQUIRE(ok.is_ok());
    REQUIRE(ok.unwrap() == 42);

    auto error_return1 = []() -> PtensorResult<int> {
        return Err(PtensorError::InvalidArgument, "invalid argument");
    };

    auto err = error_return1();
    REQUIRE(!err.is_ok());
    REQUIRE(err.unwrap_err().code() == PtensorError::InvalidArgument);

    err = Err(PtensorError::InvalidOperation, "invalid operation");
    REQUIRE(!err.is_ok());
    REQUIRE(err.unwrap_err().code() == PtensorError::InvalidOperation);
}

}  // namespace p10
