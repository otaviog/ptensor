#include <mutex>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "../spin_lock.hpp"

namespace p10::tlog {

TEST_CASE("tlog::SpinLock excludes other holders", "[tlog][spinlock]") {
    SpinLock lock;

    lock.lock();
    REQUIRE_FALSE(lock.try_lock());

    lock.unlock();
    REQUIRE(lock.try_lock());

    lock.unlock();
}

TEST_CASE("tlog::SpinLock serializes concurrent increments", "[tlog][spinlock]") {
    constexpr int THREAD_COUNT = 4;
    constexpr int INCREMENTS = 10000;

    SpinLock lock;
    // Not atomic on purpose: only the lock keeps the increments from racing.
    int counter = 0;

    std::vector<std::thread> workers;
    for (int i = 0; i < THREAD_COUNT; i++) {
        workers.emplace_back([&lock, &counter] {
            for (int increment = 0; increment < INCREMENTS; increment++) {
                const std::lock_guard<SpinLock> guard(lock);
                counter++;
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    REQUIRE(counter == THREAD_COUNT * INCREMENTS);
}

}  // namespace p10::tlog
