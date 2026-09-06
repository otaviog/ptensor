#include <algorithm>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/trompeloeil.hpp>
#include <ptensor/p10_error.hpp>
#include <ptensor/shape.hpp>
#include <ptensor/tensor.hpp>
#include <ptensor/testing/catch2_assertions.hpp>

#include "../session.hpp"
#include "mock_connection.hpp"

namespace p10::tlog {

using tests::as_text;
using tests::MockConnection;
using trompeloeil::_;

namespace {
    // Builds a session over a mocked connection, keeping a borrowed pointer
    // to it so the test can set expectations. The session owns the mock, so
    // it must outlive every expectation set on it.
    struct SessionFixture {
        SessionFixture() {
            auto connection = std::make_unique<MockConnection>();
            mock = connection.get();
            session = std::make_unique<Session>(std::move(connection));
        }

        MockConnection* mock = nullptr;
        std::unique_ptr<Session> session;
    };

    constexpr const char* ADDRESS = "127.0.0.1:4449";

    Tensor make_tensor();
}  // namespace

TEST_CASE("tlog::Session::start connects and announces the session", "[tlog][session]") {
    SessionFixture fixture;
    std::string handshake;

    trompeloeil::sequence seq;
    REQUIRE_CALL(*fixture.mock, connect(ADDRESS)).RETURN(P10Error::Ok).IN_SEQUENCE(seq);
    REQUIRE_CALL(*fixture.mock, send(_))
        .LR_SIDE_EFFECT(handshake = as_text(_1))
        .RETURN(P10Error::Ok)
        .IN_SEQUENCE(seq);

    REQUIRE_THAT(fixture.session->start(ADDRESS), testing::is_ok());

    REQUIRE(handshake.starts_with(R"({"sessionId":")"));
    // Newline delimited like the entries, so the server can read it as a line.
    REQUIRE(handshake.ends_with("\"}\n"));

    // A version 4 UUID: 32 hexadecimal digits plus 4 dashes.
    const size_t id_start = std::string(R"({"sessionId":")").size();
    REQUIRE(handshake.size() == id_start + 36 + 3);
}

TEST_CASE("tlog::Session::start uses a fresh session id", "[tlog][session]") {
    std::vector<std::string> handshakes;

    for (int i = 0; i < 2; i++) {
        SessionFixture fixture;
        ALLOW_CALL(*fixture.mock, connect(_)).RETURN(P10Error::Ok);
        REQUIRE_CALL(*fixture.mock, send(_))
            .LR_SIDE_EFFECT(handshakes.emplace_back(as_text(_1)))
            .RETURN(P10Error::Ok);

        REQUIRE_THAT(fixture.session->start(ADDRESS), testing::is_ok());
    }

    REQUIRE(handshakes[0] != handshakes[1]);
}

TEST_CASE("tlog::Session::start reports a connection failure", "[tlog][session]") {
    SessionFixture fixture;

    REQUIRE_CALL(*fixture.mock, connect(ADDRESS)).RETURN(P10Error::IoError);
    FORBID_CALL(*fixture.mock, send(_));

    REQUIRE_THAT(fixture.session->start(ADDRESS), testing::is_error(P10Error::IoError));
}

TEST_CASE("tlog::Session::start reports a handshake failure", "[tlog][session]") {
    SessionFixture fixture;

    REQUIRE_CALL(*fixture.mock, connect(ADDRESS)).RETURN(P10Error::Ok);
    REQUIRE_CALL(*fixture.mock, send(_)).RETURN(P10Error::IoError);

    REQUIRE_THAT(fixture.session->start(ADDRESS), testing::is_error(P10Error::IoError));
}

TEST_CASE("tlog::Session::log_sync sends the name and the tensor", "[tlog][session]") {
    SessionFixture fixture;
    std::vector<std::string> payloads;

    ALLOW_CALL(*fixture.mock, connect(_)).RETURN(P10Error::Ok);
    ALLOW_CALL(*fixture.mock, send(_))
        .LR_SIDE_EFFECT(payloads.emplace_back(as_text(_1)))
        .RETURN(P10Error::Ok);

    REQUIRE_THAT(fixture.session->start(ADDRESS), testing::is_ok());
    fixture.session->log_sync("input", make_tensor());

    REQUIRE(payloads.size() == 2);
    REQUIRE(payloads.back().find(R"("name")") != std::string::npos);
    REQUIRE(payloads.back().find("input") != std::string::npos);
    REQUIRE(payloads.back().find(R"("tensor")") != std::string::npos);
    REQUIRE(payloads.back().find(R"("dtype":"float32")") != std::string::npos);

    // Entries are newline delimited so the server can frame the stream.
    REQUIRE(payloads.back().ends_with('\n'));
}

TEST_CASE("tlog::Session::log_sync keeps its buffer reusable", "[tlog][session]") {
    SessionFixture fixture;
    std::vector<std::string> payloads;

    ALLOW_CALL(*fixture.mock, connect(_)).RETURN(P10Error::Ok);
    ALLOW_CALL(*fixture.mock, send(_))
        .LR_SIDE_EFFECT(payloads.emplace_back(as_text(_1)))
        .RETURN(P10Error::Ok);

    REQUIRE_THAT(fixture.session->start(ADDRESS), testing::is_ok());
    fixture.session->log_sync("first", make_tensor());
    fixture.session->log_sync("second", make_tensor());

    REQUIRE(payloads.size() == 3);
    REQUIRE(payloads[1].find("first") != std::string::npos);
    REQUIRE(payloads[2].find("second") != std::string::npos);
    REQUIRE(payloads[2].find("first") == std::string::npos);
}

TEST_CASE("tlog::Session::log_sync serializes concurrent callers", "[tlog][session]") {
    constexpr int THREAD_COUNT = 4;
    constexpr int ENTRIES_PER_THREAD = 25;

    SessionFixture fixture;
    std::vector<std::string> payloads;

    ALLOW_CALL(*fixture.mock, connect(_)).RETURN(P10Error::Ok);
    ALLOW_CALL(*fixture.mock, send(_))
        .LR_SIDE_EFFECT(payloads.emplace_back(as_text(_1)))
        .RETURN(P10Error::Ok);

    REQUIRE_THAT(fixture.session->start(ADDRESS), testing::is_ok());

    // Catch2 and trompeloeil macros only run on the main thread, so the
    // workers just collect what log_sync returned.
    std::vector<P10Error> errors(THREAD_COUNT);
    std::vector<std::thread> writers;
    for (int thread_index = 0; thread_index < THREAD_COUNT; thread_index++) {
        writers.emplace_back([&fixture, &errors, thread_index] {
            for (int entry = 0; entry < ENTRIES_PER_THREAD; entry++) {
                const std::string name = std::to_string(thread_index) + "." + std::to_string(entry);
                if (auto err = fixture.session->log_sync(name, make_tensor()); err.is_error()) {
                    errors[static_cast<size_t>(thread_index)] = err;
                }
            }
        });
    }
    for (auto& writer : writers) {
        writer.join();
    }

    for (const auto& error : errors) {
        REQUIRE_THAT(error, testing::is_ok());
    }

    // The handshake plus one payload per entry, none of them interleaved.
    REQUIRE(payloads.size() == 1 + THREAD_COUNT * ENTRIES_PER_THREAD);
    for (size_t i = 1; i < payloads.size(); i++) {
        const std::string& payload = payloads[i];
        INFO("payload " << i << ": " << payload.substr(0, 80));
        REQUIRE(payload.starts_with(R"({"name":")"));
        REQUIRE(payload.ends_with("}\n"));
        REQUIRE(std::count(payload.begin(), payload.end(), '\n') == 1);
    }
}

TEST_CASE("tlog::Session::log_sync reports a send failure", "[tlog][session]") {
    SessionFixture fixture;

    REQUIRE_CALL(*fixture.mock, send(_)).RETURN(P10Error::IoError);

    REQUIRE_THAT(
        fixture.session->log_sync("input", make_tensor()),
        testing::is_error(P10Error::IoError)
    );
}

namespace {
    Tensor make_tensor() {
        return Tensor::from_range(make_shape(2, 3)).expect("Could not create tensor");
    }
}  // namespace

}  // namespace p10::tlog
