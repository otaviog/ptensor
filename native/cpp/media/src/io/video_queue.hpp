#pragma once

#include <mutex>
#include <queue>

#include <condition_variable>

#include "video_frame.hpp"

namespace p10::media {

class VideoQueue {
  public:
    enum EmplaceResult { Ok, Cancelled };

    VideoQueue(size_t max_size) : max_size_(max_size) {}

    EmplaceResult emplace(VideoFrame&& frame) {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this]() { return queue_.size() < max_size_ || cancel_; });
        if (cancel_) {
            return EmplaceResult::Cancelled;
        }
        queue_.emplace(std::move(frame));
        cv_.notify_all();
        return EmplaceResult::Ok;
    }

    std::optional<VideoFrame> wait_and_pop() {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this]() { return !queue_.empty() || cancel_; });
        if (cancel_) {
            return std::nullopt;
        }
        VideoFrame frame = std::move(queue_.front());
        queue_.pop();
        cv_.notify_all();
        return frame;
    }

    std::optional<VideoFrame> try_pop() {
        std::unique_lock lock(mutex_);
        if (cancel_) {
            return std::nullopt;
        }
        if (queue_.empty()) {
            return std::nullopt;
        }
        VideoFrame frame = std::move(queue_.front());
        queue_.pop();
        cv_.notify_all();
        return frame;
    }

    void cancel() {
        cancel_ = true;
        cv_.notify_all();
    }

    bool empty() {
        std::unique_lock lock(mutex_);
        return queue_.empty();
    }

  private:
    size_t max_size_ = 10;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<VideoFrame> queue_;
    std::atomic<bool> cancel_ = false;
};
}  // namespace p10::media