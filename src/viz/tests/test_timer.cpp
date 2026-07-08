#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <ptensor/viz/timer.hpp>

class TestClock {
  public:
    using time_point = std::chrono::system_clock::time_point;
    using duration = std::chrono::system_clock::duration;
    using rep = std::chrono::system_clock::rep;
    using period = std::chrono::system_clock::period;

    static time_point now() {
        return current_time;
    }

    static void advance(std::chrono::milliseconds duration) {
        current_time += duration;
    }

  private:
    inline static time_point current_time = time_point {std::chrono::milliseconds {0}};
};

namespace p10::viz {
TEST_CASE("Test Timer Functionality", "[timer]") {
    constexpr p10::media::Rational BASE_TIME {4, 48000};
    Timer<TestClock> timer {};

    REQUIRE(timer.elapsed(BASE_TIME).stamp() == 0);

    timer.start();
    TestClock::advance(std::chrono::milliseconds(1000));
    auto elapsed = timer.elapsed(BASE_TIME);

    REQUIRE(elapsed.stamp() == 12000);
    REQUIRE(elapsed.base() == BASE_TIME);  // 1 second at 4/48kHz should be 16000 ticks
    REQUIRE(elapsed.to_seconds() == Catch::Approx(1.0));

    TestClock::advance(std::chrono::milliseconds(500));
    timer.pause();
    auto paused_elapsed = timer.elapsed(BASE_TIME);
    REQUIRE(paused_elapsed.stamp() == 18000);  //

    TestClock::advance(std::chrono::milliseconds(500));  // This should not count

    timer.start();
    TestClock::advance(std::chrono::milliseconds(500));
    auto final_elapsed = timer.elapsed(BASE_TIME);

    REQUIRE(final_elapsed.stamp() == 24000);
    REQUIRE(final_elapsed.to_seconds() == Catch::Approx(2.0));
}
}  // namespace p10::viz
