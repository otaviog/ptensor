#include <filesystem>
#include <fstream>

#include <catch2/catch_test_macros.hpp>
#include <ptensor/media/io/media_capture.hpp>
#include <ptensor/media/video_frame.hpp>
#include <ptensor/testing/catch2_assertions.hpp>
#include "time/rational.hpp"
#include "video_parameters.hpp"

namespace p10::media {

TEST_CASE("media::MediaCapture::open_file returns IoError for non-existent file", "[media][capture]") {
    REQUIRE_THAT(
        MediaCapture::open_file("non_existent_file.mp4"),
        testing::IsError(P10Error::IoError)
    );
}

TEST_CASE("media::MediaCapture::open_file returns IoError for unsupported file format", "[media][capture]") {
    const std::string UNSUPPORTED_FILE = "unsupported_file.txt";
    std::fstream outfile(UNSUPPORTED_FILE, std::ios::out);
    outfile << "This is not a valid media file.";
    outfile.close();

    REQUIRE_THAT(
        MediaCapture::open_file(UNSUPPORTED_FILE),
        testing::IsError(P10Error::IoError)
    );

    std::filesystem::remove(UNSUPPORTED_FILE);
}

TEST_CASE("media::MediaCapture::reads frames and parameters", "[media][capture]") {
    const std::string VALID_FILE = "tests/data/video/file_example_MP4_480_1_5MG.mp4";
    MediaCapture capture = MediaCapture::open_file(VALID_FILE).expect("should open valid file");

    SECTION("reads a single video frame") {
        REQUIRE_THAT(capture.next_frame(), testing::IsOk());

        VideoFrame frame;
        REQUIRE_THAT(capture.get_video(frame), testing::IsOk());
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

}  // namespace p10::media
