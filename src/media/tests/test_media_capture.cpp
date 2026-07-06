#include <filesystem>
#include <fstream>

#include <catch2/catch_test_macros.hpp>
#include <ptensor/media/io/media_capture.hpp>
#include <ptensor/media/io/media_device.hpp>
#include <ptensor/media/video_frame.hpp>
#include <ptensor/testing/catch2_assertions.hpp>

#include "time/rational.hpp"
#include "video_parameters.hpp"

namespace p10::media {

TEST_CASE(
    "media::MediaCapture::open_file returns IoError for non-existent file",
    "[media][capture]"
) {
    REQUIRE_THAT(
        MediaCapture::open_file("non_existent_file.mp4"),
        testing::is_error(P10Error::IoError)
    );
}

TEST_CASE(
    "media::MediaCapture::open_file returns IoError for unsupported file format",
    "[media][capture]"
) {
    const std::string unsupported_file = "unsupported_file.txt";
    std::fstream outfile(unsupported_file, std::ios::out);
    outfile << "This is not a valid media file.";
    outfile.close();

    REQUIRE_THAT(MediaCapture::open_file(unsupported_file), testing::is_error(P10Error::IoError));

    std::filesystem::remove(unsupported_file);
}

TEST_CASE("media::MediaCapture::reads frames and parameters", "[media][capture]") {
    const std::string valid_file = "tests/data/video/file_example_MP4_480_1_5MG.mp4";
    MediaCapture capture = MediaCapture::open_file(valid_file).expect("should open valid file");

    SECTION("reads a single video frame") {
        const auto result =
            capture.next_frame(MediaCapture::WaitMode::Block).expect("next_frame failed");
        REQUIRE(result == MediaCapture::Available);

        VideoFrame frame;
        REQUIRE_THAT(capture.get_video(frame), testing::is_ok());
        REQUIRE(frame.width() == 480);
        REQUIRE(frame.height() == 270);
    }

    SECTION("reads video parameters correctly") {
        const VideoParameters params = capture.get_parameters().video_parameters();
        REQUIRE(params.width() == 480);
        REQUIRE(params.height() == 270);
        REQUIRE(params.frame_rate() == Rational {30, 1});
    }
}

TEST_CASE("media::MediaCapture::open_stream rejects no device selected", "[media][capture]") {
    REQUIRE_THAT(
        MediaCapture::open_stream(std::nullopt, std::nullopt),
        testing::is_error(P10Error::InvalidArgument)
    );
}

TEST_CASE("media::MediaCapture::list_video_devices enumerates without error", "[media][capture]") {
    // We cannot assume any device is present in CI, but enumeration itself must
    // succeed (an empty list is valid). Backends that lack a programmatic
    // listing API return an empty list rather than an error.
    auto result = MediaCapture::list_video_devices();
    REQUIRE_THAT(result, testing::is_ok());

    for (const auto& device : result.unwrap()) {
        REQUIRE(device.index() >= 0);
    }
}

TEST_CASE("media::MediaCapture::list_audio_devices enumerates without error", "[media][capture]") {
    auto result = MediaCapture::list_audio_devices();
    REQUIRE_THAT(result, testing::is_ok());

    for (const auto& device : result.unwrap()) {
        REQUIRE(device.index() >= 0);
    }
}

TEST_CASE(
    "media::VideoDeviceInfo::match_closest returns default for empty capabilities",
    "[media][capture]"
) {
    VideoDeviceInfo info;
    const VideoParameters result = info.match_closest(1920, 1080, {30, 1});
    REQUIRE(result.width() == 0);
    REQUIRE(result.height() == 0);
}

TEST_CASE("media::VideoDeviceInfo::match_closest selects nearest resolution", "[media][capture]") {
    VideoDeviceInfo info;
    info.add_capability(VideoCapability {}.width(640).height(480).max_frame_rate({30, 1}));
    info.add_capability(VideoCapability {}.width(1280).height(720).max_frame_rate({30, 1}));
    info.add_capability(VideoCapability {}.width(1920).height(1080).max_frame_rate({30, 1}));

    const VideoParameters result = info.match_closest(1280, 720, {30, 1});
    REQUIRE(result.width() == 1280);
    REQUIRE(result.height() == 720);
}

TEST_CASE(
    "media::AudioDeviceInfo::match_closest returns default for empty capabilities",
    "[media][capture]"
) {
    AudioDeviceInfo info;
    const AudioParameters result = info.match_closest(44100.0, 2);
    REQUIRE(result.audio_sample_rate_hz() == 0.0);
}

TEST_CASE(
    "media::VideoDeviceInfo::match_closest picks closest when no exact match",
    "[media][capture]"
) {
    VideoDeviceInfo info;
    info.add_capability(VideoCapability {}.width(640).height(480).max_frame_rate({30, 1}));
    info.add_capability(VideoCapability {}.width(1920).height(1080).max_frame_rate({30, 1}));

    // 800x600 is closer to 640x480 than to 1920x1080 in Euclidean distance
    const VideoParameters result = info.match_closest(800, 600, {30, 1});
    REQUIRE(result.width() == 640);
    REQUIRE(result.height() == 480);
}

TEST_CASE("media::MediaCapture::WaitMode::Block never returns NotReady", "[media][capture]") {
    const std::string valid_file = "tests/data/video/file_example_MP4_480_1_5MG.mp4";
    MediaCapture capture = MediaCapture::open_file(valid_file).expect("should open valid file");

    for (int i = 0; i < 5; ++i) {
        const auto result =
            capture.next_frame(MediaCapture::WaitMode::Block).expect("next_frame failed");
        if (result == MediaCapture::Done) {
            break;
        }
        REQUIRE(result != MediaCapture::NotReady);
    }
}

TEST_CASE(
    "media::MediaCapture camera controls report NotImplemented for file captures",
    "[media][capture]"
) {
    const std::string valid_file = "tests/data/video/file_example_MP4_480_1_5MG.mp4";
    MediaCapture capture = MediaCapture::open_file(valid_file).expect("should open valid file");

    REQUIRE_THAT(capture.set_auto_focus(true), testing::is_error(P10Error::NotImplemented));
    REQUIRE_THAT(capture.get_auto_focus(), testing::is_error(P10Error::NotImplemented));
    REQUIRE_THAT(capture.set_focus_distance(10), testing::is_error(P10Error::NotImplemented));
    REQUIRE_THAT(capture.get_focus_distance(), testing::is_error(P10Error::NotImplemented));
    REQUIRE_THAT(capture.get_focus_distance_range(), testing::is_error(P10Error::NotImplemented));

    REQUIRE_THAT(capture.set_auto_exposure(true), testing::is_error(P10Error::NotImplemented));
    REQUIRE_THAT(capture.get_auto_exposure(), testing::is_error(P10Error::NotImplemented));
    REQUIRE_THAT(capture.set_exposure(10), testing::is_error(P10Error::NotImplemented));
    REQUIRE_THAT(capture.get_exposure(), testing::is_error(P10Error::NotImplemented));
    REQUIRE_THAT(capture.get_exposure_range(), testing::is_error(P10Error::NotImplemented));

    REQUIRE_THAT(capture.set_brightness(10), testing::is_error(P10Error::NotImplemented));
    REQUIRE_THAT(capture.get_brightness(), testing::is_error(P10Error::NotImplemented));
    REQUIRE_THAT(capture.get_brightness_range(), testing::is_error(P10Error::NotImplemented));

    REQUIRE_THAT(capture.set_contrast(10), testing::is_error(P10Error::NotImplemented));
    REQUIRE_THAT(capture.get_contrast(), testing::is_error(P10Error::NotImplemented));
    REQUIRE_THAT(capture.get_contrast_range(), testing::is_error(P10Error::NotImplemented));

    REQUIRE_THAT(capture.set_saturation(10), testing::is_error(P10Error::NotImplemented));
    REQUIRE_THAT(capture.get_saturation(), testing::is_error(P10Error::NotImplemented));
    REQUIRE_THAT(capture.get_saturation_range(), testing::is_error(P10Error::NotImplemented));

    REQUIRE_THAT(capture.set_gain(10), testing::is_error(P10Error::NotImplemented));
    REQUIRE_THAT(capture.get_gain(), testing::is_error(P10Error::NotImplemented));
    REQUIRE_THAT(capture.get_gain_range(), testing::is_error(P10Error::NotImplemented));

    REQUIRE_THAT(capture.set_auto_white_balance(true), testing::is_error(P10Error::NotImplemented));
    REQUIRE_THAT(capture.get_auto_white_balance(), testing::is_error(P10Error::NotImplemented));
    REQUIRE_THAT(
        capture.set_white_balance_temperature(10),
        testing::is_error(P10Error::NotImplemented)
    );
    REQUIRE_THAT(
        capture.get_white_balance_temperature(),
        testing::is_error(P10Error::NotImplemented)
    );
    REQUIRE_THAT(
        capture.get_white_balance_temperature_range(),
        testing::is_error(P10Error::NotImplemented)
    );

    REQUIRE_THAT(capture.set_zoom(10), testing::is_error(P10Error::NotImplemented));
    REQUIRE_THAT(capture.get_zoom(), testing::is_error(P10Error::NotImplemented));
    REQUIRE_THAT(capture.get_zoom_range(), testing::is_error(P10Error::NotImplemented));
}

TEST_CASE("media::MediaCapture::seek rewinds to beginning", "[media][capture]") {
    const std::string valid_file = "tests/data/video/file_example_MP4_480_1_5MG.mp4";
    MediaCapture capture = MediaCapture::open_file(valid_file).expect("should open valid file");

    // Advance several frames
    for (int i = 0; i < 10; ++i) {
        const auto result =
            capture.next_frame(MediaCapture::WaitMode::Block).expect("next_frame failed");
        if (result == MediaCapture::Done) {
            break;
        }
    }
    VideoFrame frame_before;
    REQUIRE_THAT(capture.get_video(frame_before), testing::is_ok());
    const double ts_before = frame_before.time().to_seconds();

    // Seek to start
    REQUIRE_THAT(capture.seek(0.0), testing::is_ok());

    // First frame after seek must be earlier than frame_before
    const auto result =
        capture.next_frame(MediaCapture::WaitMode::Block).expect("next_frame after seek failed");
    REQUIRE(result == MediaCapture::Available);
    VideoFrame frame_after;
    REQUIRE_THAT(capture.get_video(frame_after), testing::is_ok());
    REQUIRE(frame_after.time().to_seconds() < ts_before);
}

}  // namespace p10::media
