#include <catch2/catch_test_macros.hpp>
#include <ptensor/p10_error.hpp>
#include <ptensor/p10_result.hpp>

namespace p10 {

TEST_CASE("PtensorError::(to_string() and code())", "[error]") {
    P10Error error = P10Error::InvalidOperation;
    REQUIRE(error.to_string() == "Invalid operation");
    REQUIRE(error.code() == P10Error::InvalidOperation);

    error = P10Error::InvalidArgument;
    REQUIRE(error.to_string() == "Invalid argument");
    REQUIRE(error.code() == P10Error::InvalidArgument);

    error = P10Error::NotImplemented;
    REQUIRE(error.to_string() == "Not implemented");
    REQUIRE(error.code() == P10Error::NotImplemented);
}

TEST_CASE("PtensorError::fromAssert()", "[error]") {
    P10Error error = P10Error::fromAssert("assertion failed", "file.cpp", 42);
    REQUIRE(error.code() == P10Error::AssertionError);
    REQUIRE(error.to_string() == "Assertion error: assertion failed (file.cpp:42)");
}

TEST_CASE("PtensorResult::(Ok and Err)", "[result]") {
    P10Result<int> ok = Ok<int>(42);
    REQUIRE(ok.is_ok());
    REQUIRE(ok.unwrap() == 42);

    auto error_return1 = []() -> P10Result<int> {
        return Err(P10Error::InvalidArgument, "invalid argument");
    };

    auto err = error_return1();
    REQUIRE(!err.is_ok());
    REQUIRE(err.unwrap_err().code() == P10Error::InvalidArgument);

    err = Err(P10Error::InvalidOperation, "invalid operation");
    REQUIRE(!err.is_ok());
    REQUIRE(err.unwrap_err().code() == P10Error::InvalidOperation);
}

}  // namespace p10
