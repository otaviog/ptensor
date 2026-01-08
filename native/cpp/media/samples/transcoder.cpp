#include <iostream>

#include <ptensor/media/io/media_capture.hpp>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <media_file_path>\n";
        return 1;
    }

    auto capture = p10::media::MediaCapture::open_file(argv[1]).expect("Failed to open media file");

    const auto media_params = capture.get_parameters();
    std::cout << "Video size (width)x(height): " << media_params.video_parameters().width() << "x"
              << media_params.video_parameters().height() << "\n";

    p10::media::VideoFrame frame;
    while (capture.next_frame().expect("Failed to get next frame")) {
        capture.get_video(frame).expect("Failed to get video frame");
        std::cout << "Got video frame at: " << frame.time().to_seconds() << "s\n";

        // TODO: output the frame to other file
    }

    return 0;
}
