#pragma once

#include <atomic>

namespace p10::tlog {

/// Busy waiting lock, meant for sections that are short and rarely contended.
///
/// Preferred over `std::mutex` on the logging path because it never enters the
/// kernel: a debugger that calls into the log while it has the process stopped
/// cannot get stuck on a futex owned by a suspended thread.
///
/// Satisfies `Lockable`, so use it through `std::lock_guard<SpinLock>`.
class SpinLock {
  public:
    SpinLock() = default;
    ~SpinLock() = default;

    SpinLock(const SpinLock&) = delete;
    SpinLock(SpinLock&&) = delete;
    SpinLock& operator=(const SpinLock&) = delete;
    SpinLock& operator=(SpinLock&&) = delete;

    void lock() {
        while (locked_flag_.exchange(true, std::memory_order_acquire)) {
            // Spin on a plain load: it stays in the local cache line until the
            // owner releases, instead of fighting for exclusive access.
            while (locked_flag_.load(std::memory_order_relaxed)) {}
        }
    }

    /// Takes the lock when it is free. Lets a caller that must not block, such
    /// as one driven by a debugger, give up instead of spinning forever.
    bool try_lock() {
        return !locked_flag_.exchange(true, std::memory_order_acquire);
    }

    void unlock() {
        locked_flag_.store(false, std::memory_order_release);
    }

  private:
    std::atomic<bool> locked_flag_ {false};
};

}  // namespace p10::tlog
