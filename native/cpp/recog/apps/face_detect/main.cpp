#include <CLI/CLI.hpp>
#include <filesystem>
#include <iostream>
#include <set>
#include <string>

#include <ptensor/infer/infer.hpp>
#include <ptensor/infer/infer_config.hpp>
#include <ptensor/io/image.hpp>
#include <ptensor/media/io/media_capture.hpp>
#include <ptensor/media/video_frame.hpp>
#include <ptensor/recog/face_detection.hpp>
#include <ptensor/tensor.hpp>

using namespace p10;
using namespace p10::recog;
using namespace p10::media;

namespace {

const std::set<std::string> IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"};
const std::set<std::string> VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"};

std::string lowercase_ext(const std::string& path) {
    std::string ext = std::filesystem::path(path).extension().string();
    for (auto& c : ext) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return ext;
}

bool is_video(const std::string& path) {
    return VIDEO_EXTS.count(lowercase_ext(path)) > 0;
}

bool is_image(const std::string& path) {
    return IMAGE_EXTS.count(lowercase_ext(path)) > 0;
}

// Converts a [H, W, C] uint8 tensor to a [1, C, H, W] uint8 tensor.
P10Result<Tensor> hwc_to_nchw(const Tensor& hwc) {
    const auto src = hwc.as_span3d<uint8_t>().unwrap();
    const size_t H = src.height(), W = src.width(), C = src.channels();

    Tensor nchw;
    nchw.create(make_shape(1, int64_t(C), int64_t(H), int64_t(W)), Dtype::Uint8);

    auto* dst = reinterpret_cast<uint8_t*>(nchw.as_bytes().data());
    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                dst[c * H * W + h * W + w] = src.channel(h, w)[c];
            }
        }
    }
    return Ok<Tensor>(std::move(nchw));
}

void print_detections(const std::string& source, const FaceDetection& det) {
    for (size_t i = 0; i < det.faces.size(); i++) {
        const auto& rect = det.faces[i];
        const int x = rect.min.x;
        const int y = rect.min.y;
        const int w = rect.max.x - rect.min.x;
        const int h = rect.max.y - rect.min.y;
        const float conf = det.confidences[i];

        std::cout << source << " " << x << " " << y << " " << w << " " << h << " " << conf;

        if (i < det.landmarks.size()) {
            for (const auto& pt : det.landmarks[i]) {
                std::cout << " " << pt.x << " " << pt.y;
            }
        }
        std::cout << "\n";
    }
}

P10Error run_on_image(IFaceDetector& detector, const std::string& path) {
    auto img_result = io::load_image(path);
    if (img_result.is_error()) {
        std::cerr << "Error loading image " << path << ": " << img_result.error().to_string()
                  << "\n";
        return img_result.unwrap_err();
    }

    auto nchw_result = hwc_to_nchw(img_result.unwrap());
    if (nchw_result.is_error()) {
        return nchw_result.unwrap_err();
    }
    Tensor nchw = nchw_result.unwrap();

    std::array<FaceDetection, 1> detections;
    P10_RETURN_IF_ERROR(detector.detect(nchw, detections));

    print_detections(std::filesystem::path(path).filename().string(), detections[0]);
    return P10Error::Ok;
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

    while (true) {
        auto has_frame = cap.next_frame();
        if (has_frame.is_error() || !has_frame.unwrap()) break;

        if (auto err = cap.get_video(frame); err.is_error()) {
            std::cerr << "Error reading frame " << frame_idx << ": " << err.to_string() << "\n";
            break;
        }

        auto nchw_result = hwc_to_nchw(frame.image());
        if (nchw_result.is_error()) {
            return nchw_result.unwrap_err();
        }
        Tensor nchw = nchw_result.unwrap();

        std::array<FaceDetection, 1> detections;
        if (auto err = detector.detect(nchw, detections); err.is_error()) {
            std::cerr << "Detection error on frame " << frame_idx << ": " << err.to_string()
                      << "\n";
        } else {
            print_detections(video_name + ":" + std::to_string(frame_idx), detections[0]);
        }
        frame_idx++;
    }
    return P10Error::Ok;
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"Run face detection on images, directories, or videos"};
    argv = app.ensure_utf8(argv);

    std::string input;
    app.add_option("input", input, "Image file, image directory, or video file")->required();

    std::string model_path;
    app.add_option("--model", model_path, "ONNX model path or URL (BlazeFace)")->required();

    CLI11_PARSE(app, argc, argv);

    auto infer_result = infer::IInfer::from_onnx(model_path, infer::InferConfig());
    if (infer_result.is_error()) {
        std::cerr << "Failed to load model: " << infer_result.error().to_string() << "\n";
        return 1;
    }
    infer::IInfer* infer_engine = infer_result.unwrap();

    auto detector_result = IFaceDetector::create(BlazeFaceModel(), infer_engine);
    if (detector_result.is_error()) {
        std::cerr << "Failed to create face detector: " << detector_result.error().to_string()
                  << "\n";
        delete infer_engine;
        return 1;
    }
    IFaceDetector* detector = detector_result.unwrap();

    namespace fs = std::filesystem;
    P10Error err = P10Error::Ok;

    if (fs::is_directory(input)) {
        for (const auto& entry : fs::directory_iterator(input)) {
            if (!entry.is_regular_file() || !is_image(entry.path().string())) continue;
            run_on_image(*detector, entry.path().string());
        }
    } else if (is_video(input)) {
        err = run_on_video(*detector, input);
    } else {
        err = run_on_image(*detector, input);
    }

    delete detector;
    delete infer_engine;
    return err.is_error() ? 1 : 0;
}
