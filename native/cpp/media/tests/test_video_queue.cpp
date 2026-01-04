#include <catch2/catch_test_macros.hpp>
#include <ptensor/testing/catch2_assertions.hpp>

#include "io/video_queue.hpp"

namespace p10::media {
TEST_CASE("VideoQueue", "[media][capture]") {
    SECTION("should enqueue and dequeue video frames correctly") {
        constexpr size_t MAX_QUEUE_SIZE = 3;
        VideoQueue video_queue(MAX_QUEUE_SIZE);
        std::thread producer([&video_queue]() {
            for (int i = 0; i < 5; ++i) {
                VideoFrame frame;
                frame.create(640, 480, PixelFormat::RGB24);
                CHECK(video_queue.emplace(std::move(frame)) == VideoQueue::EmplaceResult::Ok);
            }
        });
        std::thread consumer([&video_queue]() {
            for (int i = 0; i < 5; ++i) {
                auto frame_opt = video_queue.wait_and_pop();
                CHECK(frame_opt.has_value());
                VideoFrame frame = std::move(frame_opt.value());
                CHECK(frame.width() == 640);
                CHECK(frame.height() == 480);
            }
        });

        producer.join();
        consumer.join();

        REQUIRE(video_queue.empty());
    }
    SECTION("it should cancel waiting when requested") {
        SECTION("pop should cancel with empty queue") {
            constexpr size_t MAX_QUEUE_SIZE = 3;
            VideoQueue video_queue(MAX_QUEUE_SIZE);

            std::thread consumer([&video_queue]() {
                const auto frame_opt = video_queue.wait_and_pop();
                CHECK(!frame_opt.has_value());
            });

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            video_queue.cancel();
            consumer.join();
        }
        SECTION("pop should cancel with non-empty queue") {
            constexpr size_t MAX_QUEUE_SIZE = 3;
            VideoQueue video_queue(MAX_QUEUE_SIZE);

            VideoFrame frame;
            frame.create(640, 480, PixelFormat::RGB24);
            video_queue.emplace(std::move(frame));

            video_queue.cancel();
            std::thread consumer([&video_queue]() {
                const auto frame_opt = video_queue.wait_and_pop();
                CHECK(!frame_opt.has_value());
            });
            consumer.join();
        }
        SECTION("emplace should cancel when queue is empty") {
            constexpr size_t MAX_QUEUE_SIZE = 2;
            VideoQueue video_queue(MAX_QUEUE_SIZE);

            video_queue.cancel();
            std::thread producer([&video_queue]() {
                VideoFrame frame;
                frame.create(640, 480, PixelFormat::RGB24);
                const auto result = video_queue.emplace(std::move(frame));
                CHECK(result == VideoQueue::EmplaceResult::Cancelled);
            });

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            producer.join();
        }

        SECTION("emplace should cancel when queue is full") {
            constexpr size_t MAX_QUEUE_SIZE = 1;
            VideoQueue video_queue(MAX_QUEUE_SIZE);

            VideoFrame frame1;
            frame1.create(640, 480, PixelFormat::RGB24);
            CHECK(video_queue.emplace(std::move(frame1)) == VideoQueue::EmplaceResult::Ok);

            video_queue.cancel();
            std::thread producer([&video_queue]() {
                VideoFrame frame2;
                frame2.create(640, 480, PixelFormat::RGB24);
                const auto result = video_queue.emplace(std::move(frame2));
                CHECK(result == VideoQueue::EmplaceResult::Cancelled);
            });

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            video_queue.cancel();
            producer.join();
        }
    }
}
}  // namespace p10::media