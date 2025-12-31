#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "timer.hpp"

class TestClock {
  public:
    using time_point = typename std::chrono::system_clock::time_point;
    using duration = typename std::chrono::system_clock::duration;
    using rep = typename std::chrono::system_clock::rep;
    using period = typename std::chrono::system_clock::period;

    static time_point now() {
        return current_time_;
    }

    static void advance(std::chrono::milliseconds duration) {
        current_time_ += duration;
    }

  private:
    inline static time_point current_time_ = time_point {std::chrono::milliseconds {0}};
};

TEST_CASE("Test Timer Functionality", "[timer]") {
    constexpr p10::media::Rational base_time {4, 48000};
    Timer<TestClock> timer {};

    REQUIRE(timer.elapsed(base_time).stamp() == 0);

    timer.start();
    TestClock::advance(std::chrono::milliseconds(1000));
    auto elapsed = timer.elapsed(base_time);

    REQUIRE(elapsed.stamp() == 12000);
    REQUIRE(elapsed.base() == base_time);  // 1 second at 4/48kHz should be 16000 ticks
    REQUIRE(elapsed.to_seconds() == Catch::Approx(1.0));

    TestClock::advance(std::chrono::milliseconds(500));
    timer.pause();
    auto paused_elapsed = timer.elapsed(base_time);
    REQUIRE(paused_elapsed.stamp() == 18000);  //

    TestClock::advance(std::chrono::milliseconds(500));  // This should not count

    timer.start();
    TestClock::advance(std::chrono::milliseconds(500));
    auto final_elapsed = timer.elapsed(base_time);

    REQUIRE(final_elapsed.stamp() == 24000);
    REQUIRE(final_elapsed.to_seconds() == Catch::Approx(2.0));
}