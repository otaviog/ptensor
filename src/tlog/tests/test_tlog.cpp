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

}  // namespace p10::tlog
