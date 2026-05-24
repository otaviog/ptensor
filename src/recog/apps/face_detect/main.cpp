#include <filesystem>
#include <iostream>
#include <memory>
#include <set>
#include <string>

#include <CLI/CLI.hpp>
#include <ptensor/infer/infer.hpp>
#include <ptensor/infer/infer_config.hpp>
#include <ptensor/io/image.hpp>
#include <ptensor/op/image_layout.hpp>
#include <ptensor/media/io/media_capture.hpp>
#include <ptensor/media/video_frame.hpp>
#include <ptensor/recog/face_detector.hpp>
#include <ptensor/tensor.hpp>
#include "ptensor/p10_error.hpp"

using namespace p10;
using namespace p10::recog;
using namespace p10::media;
namespace fs = std::filesystem;

namespace {

const std::set<std::string> IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"};
const std::set<std::string> VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"};

struct FaceDetectCli {
    std::string input;
    std::string model_path;
};

FaceDetectCli parse_args(int argc, char** argv);
bool is_image(const std::string& path);
std::string lowercase_ext(const std::string& path);
P10Error run_on_image(IFaceDetector& detector, const std::string& path);
void print_detections(const std::string& source, std::span<const FaceDetection> dets);
bool is_video(const std::string& path);
P10Error run_on_video(IFaceDetector& detector, const std::string& path);

}  // namespace

P10Error run_on_directory(IFaceDetector& detector, const std::string& path) {
    size_t image_count = 0;
    for (const auto& entry : fs::directory_iterator(path)) {
        if (!entry.is_regular_file() || !is_image(entry.path().string())) {
            continue;
        }
        P10_RETURN_IF_ERROR(run_on_image(detector, entry.path().string()));
        image_count++;
    }
    if (image_count == 0) {
        std::cerr << "Warning: No image files found in directory: " << path << "\n";
    }
    return P10Error::Ok;
}

int main(int argc, char** argv) {
    const auto cli = parse_args(argc, argv);

    auto infer_result = infer::IInfer::from_onnx(cli.model_path, infer::InferConfig());
    if (infer_result.is_error()) {
        std::cerr << "Error: Failed to load model: " << infer_result.error().to_string() << "\n";
        return 1;
    }

    auto detector_result = IFaceDetector::create(BlazeFaceModel(), infer_result.unwrap());
    if (detector_result.is_error()) {
        std::cerr << "Error: Failed to create face detector: "
                  << detector_result.error().to_string() << "\n";
        return 1;
    }
    std::unique_ptr<IFaceDetector> detector = detector_result.unwrap();


    P10Error err = P10Error::Ok;

    if (fs::is_directory(cli.input)) {
        err = run_on_directory(*detector, cli.input);
    } else if (is_video(cli.input)) {
        err = run_on_video(*detector, cli.input);
    } else {
        err = run_on_image(*detector, cli.input);
    }

    if (err.is_error()) {
        std::cerr << err.to_string() << std::endl;
        return 1;
    }
    return 0;
}

namespace {

FaceDetectCli parse_args(int argc, char** argv) {
    CLI::App app {
        "Run face detection on images, directories, or videos\n\n"
        "Processes images, videos, or directories containing images and outputs "
        "detected face bounding boxes with confidence scores and landmarks."
    };
    argv = app.ensure_utf8(argv);

    FaceDetectCli cli;

    app.add_option(
           "input",
           cli.input,
           "Input path: image file (.jpg, .png, .bmp, etc.), "
           "video file (.mp4, .avi, .mov, etc.), or directory of images"
    )
        ->required()
        ->check(CLI::ExistingPath);

    app.add_option("-m,--model", cli.model_path, "Path or URL to BlazeFace ONNX model")->required();

    app.footer(
        "Examples:\n"
        "  face_detect input.jpg -m model.onnx\n"
        "  face_detect video.mp4 --model model.onnx\n"
        "  face_detect ./images/ -m model.onnx"
    );

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        std::exit(app.exit(e));
    }

    return cli;
}

bool is_image(const std::string& path) {
    return IMAGE_EXTS.contains(lowercase_ext(path));
}

std::string lowercase_ext(const std::string& path) {
    std::string ext = std::filesystem::path(path).extension().string();
    for (auto& c : ext) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return ext;
}

P10Error run_on_image(IFaceDetector& detector, const std::string& path) {
    auto img_result = io::load_image(path);
    if (img_result.is_error()) {
        std::cerr << "Error loading image " << path << ": " << img_result.error().to_string()
                  << "\n";
        return img_result.unwrap_err();
    }

    Tensor nchw;
    p10::op::image_to_tensor(img_result.unwrap(), nchw, op::ImageToTensorOptions().unsqueeze(true));

    std::array<FaceDetection, 1> detections;
    P10_RETURN_IF_ERROR(detector.detect(nchw, detections));

    print_detections(std::filesystem::path(path).filename().string(), detections);
    return P10Error::Ok;
}

bool is_video(const std::string& path) {
    return VIDEO_EXTS.contains(lowercase_ext(path));
}

P10Error run_on_video(IFaceDetector& detector, const std::string& path) {
    auto cap_result = MediaCapture::open_file(path);
    if (cap_result.is_error()) {
        std::cerr << "Error opening video " << path << ": " << cap_result.error().to_string()
                  << "\n";
        return cap_result.unwrap_err();
    }
    MediaCapture cap = cap_result.unwrap();

    const std::string video_name = std::filesystem::path(path).filename().string();
    VideoFrame frame;
    int64_t frame_idx = 0;

    Tensor nchw;
    while (true) {
        if (auto has_frame = cap.next_frame(); has_frame.is_error() || !has_frame.unwrap()) {
            break;
        }

        if (auto err = cap.get_video(frame); err.is_error()) {
            std::cerr << "Error reading frame " << frame_idx << ": " << err.to_string() << "\n";
            break;
        }

        P10_RETURN_IF_ERROR(op::image_to_tensor(frame.image(), nchw));
        std::array<FaceDetection, 1> detections;
        if (auto err = detector.detect(nchw, detections); err.is_error()) {
            std::cerr << "Detection error on frame " << frame_idx << ": " << err.to_string()
                      << "\n";
        } else {
            print_detections(std::format("{}:{}", video_name, frame_idx), detections);
        }
        frame_idx++;
    }
    return P10Error::Ok;
}

void print_detections(const std::string& source, std::span<const FaceDetection> dets) {
    std::cout << "{\"" << source << "\": [\n";
    for (const auto &det : dets) {
        std::cout << to_string(det) << '\n';
    }
    std::cout << "]\n}";
}
}  // namespace
