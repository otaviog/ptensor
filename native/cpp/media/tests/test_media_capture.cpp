#include <filesystem>
#include <fstream>

#include <catch2/catch_test_macros.hpp>
#include <ptensor/media/io/media_capture.hpp>
#include <ptensor/testing/catch2_assertions.hpp>

namespace p10::media {
TEST_CASE("MediaCapture::error cases", "[media][capture]") {
    SECTION("Should return error for non-existent file") {
        REQUIRE_THAT(
            MediaCapture::open_file("non_existent_file.mp4"),
            testing::IsError(P10Error::IoError)
        );
    }
    SECTION("Should return error for unsupported file format") {
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
}

TEST_CASE("MediaCapture::basic functionality", "[media][capture]") {
    const std::string VALID_FILE = "tests/data/video/file_example_MP4_480_1_5MG.mp4";
    // Assuming test_video.mp4 is a valid media file present in the test directory.
    MediaCapture capture = MediaCapture::open_file(VALID_FILE).expect("should open valid file");

    SECTION("Should read frames from the media file") {
        REQUIRE_THAT(capture.next_frame(), testing::IsOk());

        VideoFrame frame;
        REQUIRE_THAT(capture.get_video(frame), testing::IsOk());
        REQUIRE(frame.width() == 480);  // Assuming frame is in HWC format
        REQUIRE(frame.height() == 270);
    }
    SECTION("Should read video parameters") {
        const VideoParameters params = capture.get_parameters().video_parameters();
        REQUIRE(params.width() == 480);
        REQUIRE(params.height() == 270);
        REQUIRE(params.frame_rate() == Rational {30, 1});
    }
}
}  // namespace p10::media
