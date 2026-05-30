#include <iostream>
#include <optional>
#include <string>
#include <utility>

#include <CLI/CLI.hpp>
#include <ptensor/media/io/media_capture.hpp>
#include <ptensor/media/video_parameters.hpp>
#include <ptensor/viz/gui_app.hpp>
#include <ptensor/viz/gui_app_parameters.hpp>
#include <ptensor/viz/video_player_component.hpp>

namespace {

struct PlayerCli {
    std::optional<std::string> file; ///< Path to media file (mutually exclusive with use_device).
    bool use_device = false;         ///< Capture from a live device instead of a file.
    int video_index = 0;             ///< Device index for video capture.
    int audio_index = -1;            ///< Device index for audio capture (-1 = none).
    bool list_devices = false;       ///< Print available devices and exit.
};

PlayerCli parse_args(int argc, char** argv);

class PlayerApp: public p10::viz::GuiApp {
  public:
    PlayerApp(p10::media::MediaCapture capture, std::string title) :
        player_(std::move(capture), *this),
        title_(std::move(title)) {}

  protected:
    void on_initialize() override {
        player_.on_initialize();
    }

    void on_render() override {
        player_.on_render();
    }

  private:
    p10::viz::VideoPlayerComponent player_;
    std::string title_;
};

}  // namespace

int main(int argc, char** argv) {
    auto cli = parse_args(argc, argv);

    if (cli.list_devices) {
        auto video_result = p10::media::MediaCapture::list_video_devices();
        auto audio_result = p10::media::MediaCapture::list_audio_devices();

        if (video_result.is_error()) {
            std::cerr << "Error listing video devices: " << video_result.error().to_string()
                      << "\n";
            return 1;
        }
        if (audio_result.is_error()) {
            std::cerr << "Error listing audio devices: " << audio_result.error().to_string()
                      << "\n";
            return 1;
        }

        std::cout << "Video devices:\n";
        for (const auto& d : video_result.unwrap()) {
            std::cout << "  [" << d.index() << "] " << d.name() << "\n";
        }
        std::cout << "Audio devices:\n";
        for (const auto& d : audio_result.unwrap()) {
            std::cout << "  [" << d.index() << "] " << d.name() << "\n";
        }
        return 0;
    }

    p10::media::MediaCapture capture = [&] {
        if (cli.use_device) {
            using VP = p10::media::VideoParameters;
            using AP = p10::media::AudioParameters;
            std::optional<std::pair<int, VP>> video = std::make_pair(cli.video_index, VP {});
            std::optional<std::pair<int, AP>> audio = cli.audio_index >= 0
                ? std::make_optional(std::make_pair(cli.audio_index, AP {}))
                : std::nullopt;
            return p10::media::MediaCapture::open_stream(video, audio)
                .expect("Failed to open device");
        }
        return p10::media::MediaCapture::open_file(*cli.file).expect("Failed to open file");
    }();

    const std::string title =
        cli.use_device ? std::format("device: {}", cli.video_index) : *cli.file;

    PlayerApp app(std::move(capture), title);
    const auto params = p10::viz::GuiAppParameters().title("player — " + title);
    if (auto err = run_app(app, params); err.is_error()) {
        std::cerr << err.to_string() << "\n";
        return 1;
    }
    return 0;
}

namespace {

PlayerCli parse_args(int argc, char** argv) {
    CLI::App app {"Play media files or capture from devices"};
    PlayerCli cli;

    app.add_option("file", cli.file, "Media file to play")->check(CLI::ExistingFile);
    app.add_flag("--device", cli.use_device, "Capture from video device");
    app.add_option("--video", cli.video_index, "Video device index")
        ->default_val(0)
        ->check(CLI::NonNegativeNumber);
    app.add_option("--audio", cli.audio_index, "Audio device index (-1 = none)")
        ->default_val(-1)
        ->check([](const std::string& value) -> std::string {
            try {
                if (std::stoi(value) < -1) {
                    return "Audio index must be >= -1";
                }
                return "";
            } catch (...) {
                return "Invalid audio index";
            }
        });
    app.add_flag("--list-devices", cli.list_devices, "List available devices and exit");

    app.footer(
        "Examples:\n"
        "  player video.mp4                             Play a media file\n"
        "  player --device --video 0 --audio 1          Capture from camera and mic\n"
        "  player --list-devices                        Show available devices"
    );

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        std::exit(app.exit(e));
    }

    if (!cli.use_device && !cli.list_devices && !cli.file.has_value()) {
        std::cerr << app.help();
        std::exit(1);
    }
    if (cli.use_device && cli.file.has_value()) {
        std::cerr << "Error: cannot specify both file and --device\n";
        std::exit(1);
    }

    return cli;
}

}  // namespace
