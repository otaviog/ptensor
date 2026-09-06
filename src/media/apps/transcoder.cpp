#include <iostream>
#include <string>

#include <CLI/CLI.hpp>
#include <ptensor/media/io/media_capture.hpp>
#include <ptensor/media/io/media_writer.hpp>
#include <ptensor/media/video_frame.hpp>

namespace {

struct TranscoderCli {
    std::string input;
    std::string output = "output.mp4";
    std::optional<float> start;
};

TranscoderCli parse_args(int argc, char** argv);

}  // namespace

int main(int argc, char** argv) {
    auto cli = parse_args(argc, argv);

    auto capture =
        p10::media::MediaCapture::open_file(cli.input).expect("Failed to open media file");

    if (cli.start.has_value()) {
        if (auto err = capture.seek(*cli.start); err.is_error()) {
            std::cerr << "Warning: seek failed: " << err.to_string() << "\n";
        }
    }

    auto writer = p10::media::MediaWriter::open_file(cli.output, capture.get_parameters())
                      .expect("Failed to open output media file");

    const auto media_params = capture.get_parameters();
    std::cout << "Video size (width)x(height): " << media_params.video_parameters().width() << "x"
              << media_params.video_parameters().height() << "\n";

    using p10::media::MediaCapture;
    p10::media::VideoFrame frame;
    while (capture.next_frame(MediaCapture::WaitMode::Block).expect("Failed to get next frame")
           == MediaCapture::Available) {
        capture.get_video(frame).expect("Failed to get video frame");
        std::cout << "Got video frame at: " << frame.time().to_seconds() << "s\n";
        writer.write_video(frame).expect("Failed to write video frame");
    }

    return 0;
}

namespace {
TranscoderCli parse_args(int argc, char** argv) {
    CLI::App app {"Transcode media files to MP4 format"};
    TranscoderCli cli;

    app.add_option("input", cli.input, "Input media file path")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("-o,--output", cli.output, "Output media file path")->default_val("output.mp4");
    app.add_option("-s,--start", cli.start, "Seek to position (seconds) before transcoding")
        ->check(CLI::PositiveNumber);

    app.footer(
        "Examples:\n"
        "  transcoder input.mov                         Transcode to output.mp4\n"
        "  transcoder input.mov -o output.mp4           Specify output path\n"
        "  transcoder input.mov -s 10.5 -o out.mp4     Start from 10.5 seconds"
    );

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        std::exit(app.exit(e));
    }

    return cli;
}
}  // namespace
