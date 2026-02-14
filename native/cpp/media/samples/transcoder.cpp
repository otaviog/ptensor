#include <iostream>

#include <ptensor/media/io/media_capture.hpp>
#include <ptensor/media/io/media_writer.hpp>
#include <ptensor/media/video_frame.hpp>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <media_file_path>\n";
        return 1;
    }

    auto capture = p10::media::MediaCapture::open_file(argv[1]).expect("Failed to open media file");
    auto writer = p10::media::MediaWriter::open_file("output.mp4", capture.get_parameters())
                      .expect("Failed to open output media file");
    const auto media_params = capture.get_parameters();
    std::cout << "Video size (width)x(height): " << media_params.video_parameters().width() << "x"
              << media_params.video_parameters().height() << "\n";

    p10::media::VideoFrame frame;
    while (capture.next_frame().expect("Failed to get next frame")) {
        capture.get_video(frame).expect("Failed to get video frame");
        std::cout << "Got video frame at: " << frame.time().to_seconds() << "s\n";

        writer.write_video(frame).expect("Failed to write video frame");
    }

    return 0;
}
