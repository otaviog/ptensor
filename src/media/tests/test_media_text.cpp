#include <filesystem>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <ptensor/media/io/media_capture.hpp>
#include <ptensor/media/io/media_writer.hpp>
#include <ptensor/media/text.hpp>
#include <ptensor/media/text_streams.hpp>
#include <ptensor/testing/catch2_assertions.hpp>
#include <ptensor/testing/output_path.hpp>

#include "media_parameters.hpp"
#include "text_parameters.hpp"
#include "time/rational.hpp"
#include "time/time.hpp"

namespace p10::media {

TEST_CASE("media::text stream write/read roundtrip", "[media][text]") {
    const auto output_dir = testing::get_output_path("media/text");
    const std::string out_file = (output_dir / "text_roundtrip.mkv").string();

    const Rational time_base {1, 1000};  // milliseconds

    struct Cue {
        std::string text;
        int64_t begin_ms;
        int64_t end_ms;
    };

    const std::vector<Cue> cues = {
        {"hr=72", 0, 1000},
        {R"({"bbox":[10,20,30,40]})", 1000, 2500},
        {"hr=75", 2500, 4000},
    };

    SECTION("write entries and read them back") {
        MediaParameters params;
        params.add_text_stream(TextParameters().language("eng"));

        MediaWriter writer =
            MediaWriter::open_file(out_file, params).expect("should open text output file");

        for (const auto& cue : cues) {
            const Text entry(
                cue.text,
                Time {time_base, cue.begin_ms},
                Time {time_base, cue.end_ms}
            );
            REQUIRE_THAT(writer.write_text(0, entry), testing::is_ok());
        }
        writer.close();

        MediaCapture capture =
            MediaCapture::open_file(out_file).expect("should reopen text output file");
        const TextStreams text_streams =
            capture.get_text_streams().expect("should read text streams");
        REQUIRE(text_streams.count() == 1);

        auto read_result = text_streams.get_text(0);
        REQUIRE_THAT(read_result, testing::is_ok());
        const std::vector<Text> entries = read_result.unwrap();
        REQUIRE(entries.size() == cues.size());

        for (size_t i = 0; i < cues.size(); ++i) {
            REQUIRE(entries[i].text() == cues[i].text);
            REQUIRE(
                entries[i].begin().to_seconds()
                == Catch::Approx(static_cast<double>(cues[i].begin_ms) / 1000.0).margin(0.01)
            );
            REQUIRE(
                entries[i].end().to_seconds()
                == Catch::Approx(static_cast<double>(cues[i].end_ms) / 1000.0).margin(0.05)
            );
        }
    }

    SECTION("find_text_at locates the entry active at a timestamp") {
        MediaParameters params;
        params.add_text_stream(TextParameters().language("eng"));

        MediaWriter writer =
            MediaWriter::open_file(out_file, params).expect("should open text output file");
        for (const auto& cue : cues) {
            writer.write_text(
                0,
                Text(cue.text, Time {time_base, cue.begin_ms}, Time {time_base, cue.end_ms})
            );
        }
        writer.close();

        MediaCapture capture =
            MediaCapture::open_file(out_file).expect("should reopen text output file");
        const TextStreams text_streams =
            capture.get_text_streams().expect("should read text streams");

        // Inside the second cue's interval [1000, 2500) ms.
        auto hit = text_streams.find_text_at(0, Time {time_base, 1500});
        REQUIRE_THAT(hit, testing::is_ok());
        const std::optional<Text> active = hit.unwrap();
        REQUIRE(active.has_value());
        REQUIRE(active.value().text() == cues[1].text);

        // Past the last cue's end: no entry active.
        auto miss = text_streams.find_text_at(0, Time {time_base, 9000});
        REQUIRE_THAT(miss, testing::is_ok());
        REQUIRE_FALSE(miss.unwrap().has_value());

        // Bad stream index is an error.
        REQUIRE_THAT(
            text_streams.find_text_at(3, Time {time_base, 0}),
            testing::is_error(P10Error::InvalidArgument)
        );
    }

    SECTION("get_parameters reports discovered text streams") {
        MediaParameters params;
        params.add_text_stream(TextParameters().language("eng"));

        MediaWriter writer =
            MediaWriter::open_file(out_file, params).expect("should open text output file");
        writer.write_text(0, Text(cues[0].text, Time {time_base, 0}, Time {time_base, 1000}));
        writer.close();

        MediaCapture capture =
            MediaCapture::open_file(out_file).expect("should reopen text output file");

        const MediaParameters read_params = capture.get_parameters();
        const std::vector<TextParameters>& text_streams = read_params.text_parameters();
        REQUIRE(text_streams.size() == 1);
        REQUIRE(text_streams[0].codec().type() == TextCodec::SubRip);
        REQUIRE(text_streams[0].language() == "eng");
    }

    SECTION("mp4 output rejects text streams with a helpful error") {
        MediaParameters params;
        params.add_text_stream(TextParameters().language("eng"));

        auto result = MediaWriter::open_file((output_dir / "text_roundtrip.mp4").string(), params);
        REQUIRE_THAT(result, testing::is_error(P10Error::InvalidArgument));
    }

    SECTION("writing to a missing text stream fails") {
        MediaParameters params;
        params.add_text_stream(TextParameters());

        MediaWriter writer =
            MediaWriter::open_file(out_file, params).expect("should open text output file");

        const Text entry("orphan", Time {time_base, 0}, Time {time_base, 100});
        REQUIRE_THAT(writer.write_text(5, entry), testing::is_error(P10Error::InvalidArgument));
    }

    SECTION("a source without text streams reports zero") {
        MediaParameters params;
        params.add_text_stream(TextParameters());
        MediaWriter writer =
            MediaWriter::open_file(out_file, params).expect("should open text output file");
        writer.write_text(0, Text("x", Time {time_base, 0}, Time {time_base, 10}));
        writer.close();

        const std::string no_text_file = "tests/data/video/file_example_MP4_480_1_5MG.mp4";
        if (std::filesystem::exists(no_text_file)) {
            MediaCapture capture =
                MediaCapture::open_file(no_text_file).expect("should open sample video");
            const TextStreams text_streams =
                capture.get_text_streams().expect("should read text streams");
            REQUIRE(text_streams.count() == 0);
            REQUIRE_THAT(text_streams.get_text(0), testing::is_error(P10Error::InvalidArgument));
        }
    }
}

}  // namespace p10::media
