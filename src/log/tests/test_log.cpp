#include <stdexcept>

#include <catch2/catch_test_macros.hpp>
#include <p10_internal/log/log.hpp>
#include <ptensor/testing/catch2_assertions.hpp>

using namespace p10;

TEST_CASE("log::scope creates ScopedLogger with the given scope", "[log]") {
    auto logger = log::scope("unit_test");
    REQUIRE(logger.scope == "unit_test");
}

TEST_CASE("log::ScopedLogger logs through info/error without throwing", "[log]") {
    auto logger = log::scope("unit_test");

    REQUIRE_NOTHROW(logger.info("hello"));
    REQUIRE_NOTHROW(logger.error("something went wrong"));

    const std::runtime_error error {"boom"};
    REQUIRE_NOTHROW(logger.error(error));
}

TEST_CASE("log::ScopedLogger info/error/debug accept std::format args", "[log]") {
    auto logger = log::scope("unit_test");

    REQUIRE_NOTHROW(logger.info("value={}", 42));
    REQUIRE_NOTHROW(logger.error("oops at {}:{}", "main", 7));
    REQUIRE_NOTHROW(logger.debug("plain message"));
    REQUIRE_NOTHROW(logger.debug("{} + {} = {}", 1, 2, 3));
}
