#include <string>

#include <catch2/catch_test_macros.hpp>
#include <ptensor/p10_error.hpp>
#include <ptensor/testing/catch2_assertions.hpp>

#include "../tcp_connection.hpp"
#include "loopback_server.hpp"

namespace p10::tlog {

TEST_CASE("tlog::TcpConnection::connect rejects a malformed address", "[tlog][tcp]") {
    TcpConnection connection;

    SECTION("Without a port") {
        REQUIRE_THAT(connection.connect("127.0.0.1"), p10::testing::is_error(P10Error::InvalidArgument));
    }

    SECTION("With a non numeric port") {
        REQUIRE_THAT(
            connection.connect("127.0.0.1:port"),
            p10::testing::is_error(P10Error::InvalidArgument)
        );
    }

    SECTION("With a port out of range") {
        REQUIRE_THAT(
            connection.connect("127.0.0.1:70000"),
            p10::testing::is_error(P10Error::InvalidArgument)
        );
    }

    SECTION("With a malformed host") {
        REQUIRE_THAT(
            connection.connect("not-an-address:4449"),
            p10::testing::is_error(P10Error::InvalidArgument)
        );
    }
}

TEST_CASE("tlog::TcpConnection::connect fails when nothing listens", "[tlog][tcp]") {
    TcpConnection connection;

    // Port 1 is privileged and unused, so the connect must fail rather than hang.
    REQUIRE(connection.connect("127.0.0.1:1").is_error());
}

TEST_CASE("tlog::TcpConnection sends to a local listener", "[tlog][tcp]") {
    testing::LoopbackServer server;
    REQUIRE(server.is_listening());

    TcpConnection connection;
    REQUIRE_THAT(connection.connect(server.address()), p10::testing::is_ok());
    REQUIRE_THAT(connection.send(std::string("first\n")), p10::testing::is_ok());
    REQUIRE_THAT(connection.send(std::string("second\n")), p10::testing::is_ok());

    // The server reads until the client goes away.
    connection.close();
    REQUIRE(server.received() == "first\nsecond\n");
}

TEST_CASE("tlog::TcpConnection rejects a second connect", "[tlog][tcp]") {
    testing::LoopbackServer server;
    REQUIRE(server.is_listening());

    TcpConnection connection;
    REQUIRE_THAT(connection.connect(server.address()), p10::testing::is_ok());
    REQUIRE_THAT(connection.connect(server.address()), p10::testing::is_error(P10Error::IoError));
}

TEST_CASE("tlog::TcpConnection::send without a connection", "[tlog][tcp]") {
    TcpConnection connection;

    REQUIRE_THAT(connection.send(std::string("payload")), p10::testing::is_error(P10Error::IoError));
}

TEST_CASE("tlog::TcpConnection::send after close", "[tlog][tcp]") {
    testing::LoopbackServer server;
    REQUIRE(server.is_listening());

    TcpConnection connection;
    REQUIRE_THAT(connection.connect(server.address()), p10::testing::is_ok());
    connection.close();

    REQUIRE_THAT(connection.send(std::string("payload")), p10::testing::is_error(P10Error::IoError));
}

}  // namespace p10::tlog
