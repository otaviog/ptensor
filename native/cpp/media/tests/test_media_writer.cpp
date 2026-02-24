#include <filesystem>
#include <fstream>

#include <catch2/catch_test_macros.hpp>
#include <ptensor/media/io/media_capture.hpp>
#include <ptensor/media/io/media_writer.hpp>
#include <ptensor/media/video_frame.hpp>
#include <ptensor/testing/catch2_assertions.hpp>

#include "catch2/catch_approx.hpp"
#include "io/media_capture.hpp"
#include "media_parameters.hpp"
#include "time/rational.hpp"
#include "video_parameters.hpp"

namespace p10::media {
TEST_CASE("media::MediaWriter::error cases", "[media][writer]") {
    SECTION("Should return error for non-existent parent directory") {
        REQUIRE_THAT(
            MediaWriter::open_file("non_existing_dir/non_existent_file.mp4", MediaParameters()),
            testing::IsError(P10Error::IoError)
        );
    }
    SECTION("Should return error for unsupported file format") {
        const std::string UNSUPPORTED_FILE = "unsupported_file.txt";
        std::fstream outfile(UNSUPPORTED_FILE, std::ios::out);
        outfile << "This is not a valid media file.";
        outfile.close();
        REQUIRE_THAT(
            MediaWriter::open_file(UNSUPPORTED_FILE, MediaParameters()),
            testing::IsError(P10Error::IoError)
        );
        std::filesystem::remove(UNSUPPORTED_FILE);
    }
}

TEST_CASE("media::MediaWriter::basic functionality", "[media][writer]") {
    const std::string VALID_FILE = "tests/data/video/file_example_MP4_480_1_5MG.mp4";
    const std::string OUT_FILE = "tests/output/file_example_MP4_480_1_5MG_out.mp4";
    const int TOTAL_FRAMES = 100;

    SECTION("Should transcode a media correctly") {
        SECTION("read and write frames from the media file correctly") {
            MediaCapture capture =
                MediaCapture::open_file(VALID_FILE).expect("should open valid file");
            MediaWriter writer = MediaWriter::open_file(OUT_FILE, capture.get_parameters())
                                     .expect("should open output file for write");

            for (int current_frame = 0; current_frame < TOTAL_FRAMES; ++current_frame) {
                REQUIRE_THAT(capture.next_frame(), testing::IsOk());

                VideoFrame video_frame;
                REQUIRE_THAT(capture.get_video(video_frame), testing::IsOk());
                writer.write_video(video_frame);
            }
        }

        SECTION("match written media with source") {
            MediaCapture capture =
                MediaCapture::open_file(VALID_FILE).expect("should open valid file");
            MediaCapture capture_out =
                MediaCapture::open_file(OUT_FILE).expect("should open output file");

            const auto out_seconds = double(TOTAL_FRAMES)
                / capture.get_parameters().video_parameters().frame_rate().to_double();
            REQUIRE(capture_out.duration() == Catch::Approx(out_seconds).margin(0.1));

            REQUIRE(capture_out.video_frame_count() == TOTAL_FRAMES);
            for (int current_frame = 0; current_frame < TOTAL_FRAMES; ++current_frame) {
                VideoFrame video_frame_in;
                REQUIRE_THAT(capture.next_frame(), testing::IsOk());
                REQUIRE_THAT(capture.get_video(video_frame_in), testing::IsOk());

                REQUIRE_THAT(capture_out.next_frame(), testing::IsOk());
                VideoFrame video_frame_out;
                REQUIRE_THAT(capture_out.get_video(video_frame_out), testing::IsOk());
                REQUIRE(video_frame_in.width() == video_frame_out.width());
                REQUIRE(video_frame_in.height() == video_frame_out.height());
            }
        }
    }

    SECTION("Should handle different parameters") {
        const Rational NEW_FRAME_RATE(24, 1);

        MediaCapture capture = MediaCapture::open_file(VALID_FILE).expect("should open valid file");
        MediaParameters original_params = capture.get_parameters();
        REQUIRE(original_params.video_parameters().frame_rate() != NEW_FRAME_RATE);

        MediaParameters new_params = original_params;

        const int new_width = original_params.video_parameters().width() / 2;
        const int new_height = (original_params.video_parameters().height() / 2) & ~1;

        new_params.video_parameters()
            .frame_rate(NEW_FRAME_RATE)
            .width(new_width)
            .height(new_height);

        MediaWriter writer = MediaWriter::open_file(OUT_FILE, new_params)
                                 .expect("should open output file for write");

        for (int current_frame = 0; current_frame < TOTAL_FRAMES; ++current_frame) {
            REQUIRE_THAT(capture.next_frame(), testing::IsOk());
            VideoFrame video_frame;
            REQUIRE_THAT(capture.get_video(video_frame), testing::IsOk());
            writer.write_video(video_frame);
        }
        writer.close();

        // check media read with different parameters
        MediaCapture capture_out =
            MediaCapture::open_file(OUT_FILE).expect("should open output file");

        MediaParameters params = capture_out.get_parameters();
        REQUIRE(params.video_parameters().width() == new_width);
        REQUIRE(params.video_parameters().height() == new_height);
        REQUIRE(params.video_parameters().frame_rate() == NEW_FRAME_RATE);
    }
}
}  // namespace p10::media
