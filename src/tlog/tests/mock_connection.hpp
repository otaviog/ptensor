#pragma once

#include <cstddef>
#include <span>
#include <string>

#include <catch2/trompeloeil.hpp>
#include <ptensor/p10_error.hpp>

#include <string_view>

#include "../connection.hpp"

namespace p10::tlog::tests {

class MockConnection: public IConnection {
  public:
    MAKE_MOCK1(connect, P10Error(const std::string&), override);
    MAKE_MOCK0(close, void(), override);
    MAKE_CONST_MOCK1(send, P10Error(std::span<const std::byte>), override);
};

/// Views a sent payload as text, so expectations can match on the JSON.
inline std::string_view as_text(std::span<const std::byte> data) {
    return {reinterpret_cast<const char*>(data.data()), data.size()};
}

}  // namespace p10::tlog::tests
