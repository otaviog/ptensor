#include <cstdlib>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <ptensor/shape.hpp>
#include <ptensor/tensor.hpp>
#include <ptensor/tlog/tlog.hpp>

#include "loopback_server.hpp"

namespace p10::tlog {

// Hidden by default: log() opens the process wide session on its first call,
// so this case has to own the endpoint for the whole run.
TEST_CASE("tlog::log streams tensors to the server", "[.tlog][.integration]") {
    testing::LoopbackServer server;
    REQUIRE(server.is_listening());
    REQUIRE(::setenv("PTENSOR_TLOG_ADDRESS", server.address().c_str(), 1) == 0);

    // One flow rather than sections: the session sticks to the first server it
    // reaches, so every entry has to go through this one.
    log("zeros", Tensor::zeros(make_shape(2, 3)).expect("Could not create tensor"));

    // The stream starts with the handshake, the entry follows it.
    const std::string handshake = server.wait_for(1);
    REQUIRE(handshake.starts_with(R"({"sessionId":")"));

    const std::string first = server.wait_for(handshake.size() + 1);
    REQUIRE(first.find(R"("name":"zeros")") != std::string::npos);
    REQUIRE(first.find(R"("dtype":"float32")") != std::string::npos);
    REQUIRE(first.ends_with('\n'));

    log("empty", Tensor());

    const std::string second = server.wait_for(first.size() + 1);
    REQUIRE(second.size() > first.size());
    REQUIRE(second.find(R"("name":"empty")") != std::string::npos);
    REQUIRE(second.ends_with('\n'));
}

// Not hidden: log_to takes its endpoint per call, so it needs no process wide
// state and cannot collide with another case.
TEST_CASE("tlog::log_to streams tensors to the given address", "[tlog][integration]") {
    testing::LoopbackServer server;
    REQUIRE(server.is_listening());

    log_to(server.address().c_str(), "zeros", Tensor::zeros(make_shape(2, 3)).expect("tensor"));

    const std::string handshake = server.wait_for(1);
    REQUIRE(handshake.starts_with(R"({"sessionId":")"));

    const std::string entry = server.wait_for(handshake.size() + 1);
    REQUIRE(entry.find(R"("name":"zeros")") != std::string::npos);
    REQUIRE(entry.ends_with('\n'));

    // The address keeps its session, so the second entry reuses the connection
    // rather than announcing a second one.
    log_to(server.address().c_str(), "ones", Tensor::zeros(make_shape(1)).expect("tensor"));

    const std::string both = server.wait_for(entry.size() + 1);
    REQUIRE(both.find(R"("name":"ones")") != std::string::npos);
    REQUIRE(both.find(R"({"sessionId":")") == both.rfind(R"({"sessionId":")"));
}

// An unreachable endpoint is logged and swallowed: logging must not change the
// caller's flow.
TEST_CASE("tlog::log_to survives an address nothing listens on", "[tlog]") {
    log_to("127.0.0.1:1", "zeros", Tensor::zeros(make_shape(2)).expect("tensor"));
}

}  // namespace p10::tlog
