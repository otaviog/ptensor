#include <catch2/catch_test_macros.hpp>
#include <ptensor/testing/catch2_assertions.hpp>

#include "io/ffmpeg/ffmpeg_sws.hpp"
#include "video_frame.hpp"

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
}

namespace p10::media {

namespace {
// Helper to create and fill an AVFrame with test data
AVFrame* create_test_avframe(int width, int height, AVPixelFormat format) {
    AVFrame* frame = av_frame_alloc();
    frame->width = width;
    frame->height = height;
    frame->format = format;

    int buffer_size = av_image_get_buffer_size(format, width, height, 1);
    uint8_t* buffer = (uint8_t*)av_malloc(buffer_size);

    // Fill with test pattern
    for (int i = 0; i < buffer_size; ++i) {
        buffer[i] = static_cast<uint8_t>(i % 256);
    }

    av_image_fill_arrays(frame->data, frame->linesize, buffer, format, width, height, 1);

    return frame;
}

void free_test_avframe(AVFrame* frame) {
    if (frame) {
        av_free(frame->data[0]);
        av_frame_free(&frame);
    }
}

VideoFrame create_test_videoframe(int width, int height) {
    VideoFrame frame;
    frame.create(width, height, PixelFormat::RGB24).expect("create video frame");

    // Fill with test pattern
    auto bytes = frame.as_bytes();
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<uint8_t>(i % 256);
    }

    return frame;
}

}  // namespace

TEST_CASE("FfmpegSws::target size getters and setters", "[media][ffmpeg][sws]") {
    FfmpegSws sws;

    SECTION("initial state has no target size") {
        REQUIRE_FALSE(sws.target_width().has_value());
        REQUIRE_FALSE(sws.target_height().has_value());
    }

    SECTION("set and get target size") {
        sws.set_target_size(640, 480);
        REQUIRE(sws.target_width().has_value());
        REQUIRE(sws.target_height().has_value());
        REQUIRE(sws.target_width().value() == 640);
        REQUIRE(sws.target_height().value() == 480);
    }

    SECTION("reset target size") {
        sws.set_target_size(640, 480);
        sws.reset_target_size();
        REQUIRE_FALSE(sws.target_width().has_value());
        REQUIRE_FALSE(sws.target_height().has_value());
    }
}

TEST_CASE("FfmpegSws::target pixel format", "[media][ffmpeg][sws]") {
    FfmpegSws sws;

    SECTION("default pixel format is RGB24") {
        REQUIRE(sws.target_pixel_format() == AV_PIX_FMT_RGB24);
    }

    SECTION("set and get target pixel format") {
        sws.set_target_pixel_format(AV_PIX_FMT_YUV420P);
        REQUIRE(sws.target_pixel_format() == AV_PIX_FMT_YUV420P);

        sws.set_target_pixel_format(AV_PIX_FMT_GRAY8);
        REQUIRE(sws.target_pixel_format() == AV_PIX_FMT_GRAY8);
    }
}

TEST_CASE("FfmpegSws::transform AVFrame to VideoFrame", "[media][ffmpeg][sws]") {
    FfmpegSws sws;

    SECTION("transform without scaling") {
        AVFrame* src_frame = create_test_avframe(320, 240, AV_PIX_FMT_YUV420P);
        VideoFrame dst_frame;

        REQUIRE_THAT(sws.transform(src_frame, dst_frame), testing::IsOk());

        REQUIRE(dst_frame.width() == 320);
        REQUIRE(dst_frame.height() == 240);
        REQUIRE(dst_frame.pixel_format() == PixelFormat::RGB24);

        free_test_avframe(src_frame);
    }

    SECTION("transform with upscaling") {
        AVFrame* src_frame = create_test_avframe(320, 240, AV_PIX_FMT_YUV420P);
        VideoFrame dst_frame;

        sws.set_target_size(640, 480);

        REQUIRE_THAT(sws.transform(src_frame, dst_frame), testing::IsOk());

        REQUIRE(dst_frame.width() == 640);
        REQUIRE(dst_frame.height() == 480);
        REQUIRE(dst_frame.pixel_format() == PixelFormat::RGB24);

        free_test_avframe(src_frame);
    }

    SECTION("transform with downscaling") {
        AVFrame* src_frame = create_test_avframe(640, 480, AV_PIX_FMT_YUV420P);
        VideoFrame dst_frame;

        sws.set_target_size(320, 240);

        REQUIRE_THAT(sws.transform(src_frame, dst_frame), testing::IsOk());

        REQUIRE(dst_frame.width() == 320);
        REQUIRE(dst_frame.height() == 240);
        REQUIRE(dst_frame.pixel_format() == PixelFormat::RGB24);

        free_test_avframe(src_frame);
    }

    SECTION("transform with aspect ratio change") {
        AVFrame* src_frame = create_test_avframe(640, 480, AV_PIX_FMT_YUV420P);
        VideoFrame dst_frame;

        sws.set_target_size(800, 600);

        REQUIRE_THAT(sws.transform(src_frame, dst_frame), testing::IsOk());

        REQUIRE(dst_frame.width() == 800);
        REQUIRE(dst_frame.height() == 600);

        free_test_avframe(src_frame);
    }

    SECTION("transform RGB24 frame without conversion") {
        AVFrame* src_frame = create_test_avframe(320, 240, AV_PIX_FMT_RGB24);
        VideoFrame dst_frame;

        REQUIRE_THAT(sws.transform(src_frame, dst_frame), testing::IsOk());

        REQUIRE(dst_frame.width() == 320);
        REQUIRE(dst_frame.height() == 240);

        free_test_avframe(src_frame);
    }
}

TEST_CASE("FfmpegSws::transform VideoFrame to AVFrame", "[media][ffmpeg][sws]") {
    FfmpegSws sws;

    SECTION("transform without scaling") {
        VideoFrame src_frame = create_test_videoframe(320, 240);
        AVFrame* dst_frame = nullptr;

        REQUIRE_THAT(sws.transform(src_frame, &dst_frame), testing::IsOk());

        REQUIRE(dst_frame != nullptr);
        REQUIRE(dst_frame->width == 320);
        REQUIRE(dst_frame->height == 240);
        REQUIRE(dst_frame->format == AV_PIX_FMT_RGB24);

        free_test_avframe(dst_frame);
    }

    SECTION("transform with upscaling") {
        VideoFrame src_frame = create_test_videoframe(320, 240);
        AVFrame* dst_frame = nullptr;

        sws.set_target_size(640, 480);

        REQUIRE_THAT(sws.transform(src_frame, &dst_frame), testing::IsOk());

        REQUIRE(dst_frame != nullptr);
        REQUIRE(dst_frame->width == 640);
        REQUIRE(dst_frame->height == 480);

        free_test_avframe(dst_frame);
    }

    SECTION("transform with downscaling") {
        VideoFrame src_frame = create_test_videoframe(640, 480);
        AVFrame* dst_frame = nullptr;

        sws.set_target_size(320, 240);

        REQUIRE_THAT(sws.transform(src_frame, &dst_frame), testing::IsOk());

        REQUIRE(dst_frame != nullptr);
        REQUIRE(dst_frame->width == 320);
        REQUIRE(dst_frame->height == 240);

        free_test_avframe(dst_frame);
    }

    SECTION("transform with pixel format conversion to YUV420P") {
        VideoFrame src_frame = create_test_videoframe(320, 240);
        AVFrame* dst_frame = nullptr;

        sws.set_target_pixel_format(AV_PIX_FMT_YUV420P);

        REQUIRE_THAT(sws.transform(src_frame, &dst_frame), testing::IsOk());

        REQUIRE(dst_frame != nullptr);
        REQUIRE(dst_frame->width == 320);
        REQUIRE(dst_frame->height == 240);
        REQUIRE(dst_frame->format == AV_PIX_FMT_YUV420P);

        free_test_avframe(dst_frame);
    }

    SECTION("transform with both scaling and format conversion") {
        VideoFrame src_frame = create_test_videoframe(320, 240);
        AVFrame* dst_frame = nullptr;

        sws.set_target_size(640, 480);
        sws.set_target_pixel_format(AV_PIX_FMT_YUV420P);

        REQUIRE_THAT(sws.transform(src_frame, &dst_frame), testing::IsOk());

        REQUIRE(dst_frame != nullptr);
        REQUIRE(dst_frame->width == 640);
        REQUIRE(dst_frame->height == 480);
        REQUIRE(dst_frame->format == AV_PIX_FMT_YUV420P);

        free_test_avframe(dst_frame);
    }
}

TEST_CASE("FfmpegSws::multiple transforms with context reuse", "[media][ffmpeg][sws]") {
    FfmpegSws sws;
    sws.set_target_size(640, 480);

    SECTION("multiple transforms reuse the context") {
        for (int i = 0; i < 5; ++i) {
            AVFrame* src_frame = create_test_avframe(320, 240, AV_PIX_FMT_YUV420P);
            VideoFrame dst_frame;

            REQUIRE_THAT(sws.transform(src_frame, dst_frame), testing::IsOk());
            REQUIRE(dst_frame.width() == 640);
            REQUIRE(dst_frame.height() == 480);

            free_test_avframe(src_frame);
        }
    }

    SECTION("changing target size recreates context") {
        AVFrame* src_frame1 = create_test_avframe(320, 240, AV_PIX_FMT_YUV420P);
        VideoFrame dst_frame1;

        REQUIRE_THAT(sws.transform(src_frame1, dst_frame1), testing::IsOk());
        REQUIRE(dst_frame1.width() == 640);
        REQUIRE(dst_frame1.height() == 480);

        free_test_avframe(src_frame1);

        // Change target size
        sws.set_target_size(800, 600);

        AVFrame* src_frame2 = create_test_avframe(320, 240, AV_PIX_FMT_YUV420P);
        VideoFrame dst_frame2;

        REQUIRE_THAT(sws.transform(src_frame2, dst_frame2), testing::IsOk());
        REQUIRE(dst_frame2.width() == 800);
        REQUIRE(dst_frame2.height() == 600);

        free_test_avframe(src_frame2);
    }
}

TEST_CASE("FfmpegSws::edge cases", "[media][ffmpeg][sws]") {
    FfmpegSws sws;

    SECTION("small frame dimensions") {
        AVFrame* src_frame = create_test_avframe(16, 16, AV_PIX_FMT_YUV420P);
        VideoFrame dst_frame;

        REQUIRE_THAT(sws.transform(src_frame, dst_frame), testing::IsOk());
        REQUIRE(dst_frame.width() == 16);
        REQUIRE(dst_frame.height() == 16);

        free_test_avframe(src_frame);
    }

    SECTION("large frame dimensions") {
        AVFrame* src_frame = create_test_avframe(1920, 1080, AV_PIX_FMT_YUV420P);
        VideoFrame dst_frame;

        sws.set_target_size(3840, 2160);

        REQUIRE_THAT(sws.transform(src_frame, dst_frame), testing::IsOk());
        REQUIRE(dst_frame.width() == 3840);
        REQUIRE(dst_frame.height() == 2160);

        free_test_avframe(src_frame);
    }

    SECTION("odd dimensions") {
        AVFrame* src_frame = create_test_avframe(321, 241, AV_PIX_FMT_YUV420P);
        VideoFrame dst_frame;

        REQUIRE_THAT(sws.transform(src_frame, dst_frame), testing::IsOk());
        REQUIRE(dst_frame.width() == 321);
        REQUIRE(dst_frame.height() == 241);

        free_test_avframe(src_frame);
    }
}

}  // namespace p10::media
