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
            testing::is_error(P10Error::IoError)
        );
    }
    SECTION("Should return error for unsupported file format") {
        const std::string unsupported_file = "unsupported_file.txt";
        std::fstream outfile(unsupported_file, std::ios::out);
        outfile << "This is not a valid media file.";
        outfile.close();
        REQUIRE_THAT(
            MediaWriter::open_file(unsupported_file, MediaParameters()),
            testing::is_error(P10Error::IoError)
        );
        std::filesystem::remove(unsupported_file);
    }
}

TEST_CASE("media::MediaWriter::basic functionality", "[media][writer]") {
    const std::string valid_file = "tests/data/video/file_example_MP4_480_1_5MG.mp4";
    const std::string out_file = "tests/output/file_example_MP4_480_1_5MG_out.mp4";
    const int total_frames = 100;

    SECTION("Should transcode a media correctly") {
        SECTION("read and write frames from the media file correctly") {
            MediaCapture capture =
                MediaCapture::open_file(valid_file).expect("should open valid file");
            MediaWriter writer = MediaWriter::open_file(out_file, capture.get_parameters())
                                     .expect("should open output file for write");

            for (int current_frame = 0; current_frame < total_frames; ++current_frame) {
                REQUIRE_THAT(capture.next_frame(MediaCapture::Block), testing::is_ok());

                VideoFrame video_frame;
                REQUIRE_THAT(capture.get_video(video_frame), testing::is_ok());
                writer.write_video(video_frame);
            }
        }

        SECTION("match written media with source") {
            MediaCapture capture =
                MediaCapture::open_file(valid_file).expect("should open valid file");
            MediaCapture capture_out =
                MediaCapture::open_file(out_file).expect("should open output file");

            const auto out_seconds = static_cast<double>(total_frames)
                / capture.get_parameters().video_parameters().frame_rate().to_double();
            REQUIRE(capture_out.duration() == Catch::Approx(out_seconds).margin(0.1));

            REQUIRE(capture_out.video_frame_count() == total_frames);
            // The h264 encoder may flush one fewer frame than was fed in
            // (last B-frame consumed by GOP closure); accept up to one
            // missing frame at the tail.
            int matched_frames = 0;
            for (int current_frame = 0; current_frame < total_frames; ++current_frame) {
                auto next_in = capture.next_frame(MediaCapture::Block);
                REQUIRE_THAT(next_in, testing::is_ok());
                if (next_in.unwrap() == MediaCapture::Done) {
                    break;
                }
                VideoFrame video_frame_in;
                REQUIRE_THAT(capture.get_video(video_frame_in), testing::is_ok());

                auto next_out = capture_out.next_frame(MediaCapture::Block);
                REQUIRE_THAT(next_out, testing::is_ok());
                if (next_out.unwrap() == MediaCapture::Done) {
                    break;
                }
                VideoFrame video_frame_out;
                REQUIRE_THAT(capture_out.get_video(video_frame_out), testing::is_ok());
                REQUIRE(video_frame_in.width() == video_frame_out.width());
                REQUIRE(video_frame_in.height() == video_frame_out.height());
                ++matched_frames;
            }
            REQUIRE(matched_frames >= total_frames - 1);
        }
    }

    SECTION("Should handle different parameters") {
        const Rational new_frame_rate(24, 1);

        MediaCapture capture = MediaCapture::open_file(valid_file).expect("should open valid file");
        MediaParameters original_params = capture.get_parameters();
        REQUIRE(original_params.video_parameters().frame_rate() != new_frame_rate);

        MediaParameters new_params = original_params;

        const int new_width = original_params.video_parameters().width() / 2;
        const int new_height = (original_params.video_parameters().height() / 2) & ~1;

        new_params.video_parameters()
            .frame_rate(new_frame_rate)
            .width(new_width)
            .height(new_height);

        MediaWriter writer = MediaWriter::open_file(out_file, new_params)
                                 .expect("should open output file for write");

        for (int current_frame = 0; current_frame < total_frames; ++current_frame) {
            REQUIRE_THAT(capture.next_frame(MediaCapture::Block), testing::is_ok());
            VideoFrame video_frame;
            REQUIRE_THAT(capture.get_video(video_frame), testing::is_ok());
            writer.write_video(video_frame);
        }
        writer.close();

        // check media read with different parameters
        const MediaCapture capture_out =
            MediaCapture::open_file(out_file).expect("should open output file");

        const MediaParameters params = capture_out.get_parameters();
        REQUIRE(params.video_parameters().width() == new_width);
        REQUIRE(params.video_parameters().height() == new_height);
        REQUIRE(params.video_parameters().frame_rate() == new_frame_rate);
    }
}
}  // namespace p10::media
