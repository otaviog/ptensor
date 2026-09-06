#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

#include <CLI/CLI.hpp>
#include <imgui.h>
#include <ptensor/media/io/media_capture.hpp>
#include <ptensor/viz/gui_app.hpp>
#include <ptensor/viz/gui_app_parameters.hpp>
#include <ptensor/viz/video_player_component.hpp>

using p10::media::CameraControlRange;
using p10::media::MediaCapture;

namespace {

struct CameraCli {
    int device_index = 0;  ///< Video device index to open.
    bool list_devices = false;  ///< Print available video devices and exit.
};

CameraCli parse_args(int argc, char** argv) {
    CLI::App app {"View a camera stream and adjust its controls (focus, exposure, ...)"};
    CameraCli cli;

    app.add_option("device", cli.device_index, "Video device index")
        ->default_val(0)
        ->check(CLI::NonNegativeNumber);
    app.add_flag("--list-devices", cli.list_devices, "List available video devices and exit");

    app.footer(
        "Examples:\n"
        "  camera_controller                 Open the default camera (device 0)\n"
        "  camera_controller 1               Open video device 1\n"
        "  camera_controller --list-devices  Print the available cameras"
    );

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        std::exit(app.exit(e));
    }

    return cli;
}

/// UI state for one integer camera control.
struct SliderControl {
    bool available = false;
    int value = 0;
    CameraControlRange range;
};

/// UI state for one auto/manual toggle.
struct AutoControl {
    bool available = false;
    bool enabled = false;
};

/// ImGui panel exposing the camera controls of a live capture. Controls that
/// the device (or platform backend) does not support are shown as disabled.
class CameraControlsPanel {
  public:
    explicit CameraControlsPanel(MediaCapture capture) : capture_(std::move(capture)) {}

    void initialize() {
        refresh();
    }

    /// Re-read every control from the device. Also needed after toggling an
    /// auto mode, since the device may move the manual value on its own.
    void refresh() {
        auto_focus_ = load_auto([&] { return capture_.get_auto_focus(); });
        focus_ = load_slider(
            [&] { return capture_.get_focus_distance_range(); },
            [&] { return capture_.get_focus_distance(); }
        );

        auto_exposure_ = load_auto([&] { return capture_.get_auto_exposure(); });
        exposure_ = load_slider(
            [&] { return capture_.get_exposure_range(); },
            [&] { return capture_.get_exposure(); }
        );

        auto_white_balance_ = load_auto([&] { return capture_.get_auto_white_balance(); });
        white_balance_ = load_slider(
            [&] { return capture_.get_white_balance_temperature_range(); },
            [&] { return capture_.get_white_balance_temperature(); }
        );

        brightness_ = load_slider(
            [&] { return capture_.get_brightness_range(); },
            [&] { return capture_.get_brightness(); }
        );
        contrast_ = load_slider(
            [&] { return capture_.get_contrast_range(); },
            [&] { return capture_.get_contrast(); }
        );
        saturation_ = load_slider(
            [&] { return capture_.get_saturation_range(); },
            [&] { return capture_.get_saturation(); }
        );
        gain_ = load_slider(
            [&] { return capture_.get_gain_range(); },
            [&] { return capture_.get_gain(); }
        );
        zoom_ = load_slider(
            [&] { return capture_.get_zoom_range(); },
            [&] { return capture_.get_zoom(); }
        );
    }

    void render() {
        ImGui::SetNextWindowSize(ImVec2(360, 480), ImGuiCond_FirstUseEver);
        ImGui::Begin("Camera Controls");

        ImGui::SeparatorText("Focus");
        auto_checkbox("Auto focus", auto_focus_, [&](bool enabled) {
            return capture_.set_auto_focus(enabled);
        });
        ImGui::BeginDisabled(auto_focus_.available && auto_focus_.enabled);
        slider("Focus distance", focus_, [&](int value) {
            return capture_.set_focus_distance(value);
        });
        ImGui::EndDisabled();

        ImGui::SeparatorText("Exposure");
        auto_checkbox("Auto exposure", auto_exposure_, [&](bool enabled) {
            return capture_.set_auto_exposure(enabled);
        });
        ImGui::BeginDisabled(auto_exposure_.available && auto_exposure_.enabled);
        slider("Exposure", exposure_, [&](int value) { return capture_.set_exposure(value); });
        ImGui::EndDisabled();
        slider("Gain", gain_, [&](int value) { return capture_.set_gain(value); });

        ImGui::SeparatorText("White balance");
        auto_checkbox("Auto white balance", auto_white_balance_, [&](bool enabled) {
            return capture_.set_auto_white_balance(enabled);
        });
        ImGui::BeginDisabled(auto_white_balance_.available && auto_white_balance_.enabled);
        slider("Temperature", white_balance_, [&](int value) {
            return capture_.set_white_balance_temperature(value);
        });
        ImGui::EndDisabled();

        ImGui::SeparatorText("Image");
        slider("Brightness", brightness_, [&](int value) {
            return capture_.set_brightness(value);
        });
        slider("Contrast", contrast_, [&](int value) { return capture_.set_contrast(value); });
        slider("Saturation", saturation_, [&](int value) {
            return capture_.set_saturation(value);
        });
        slider("Zoom", zoom_, [&](int value) { return capture_.set_zoom(value); });

        ImGui::Separator();
        if (ImGui::Button("Refresh")) {
            refresh();
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset to defaults")) {
            reset_to_defaults();
        }

        if (!last_error_.empty()) {
            ImGui::TextColored(ImVec4(1.0F, 0.4F, 0.4F, 1.0F), "%s", last_error_.c_str());
        }

        ImGui::End();
    }

  private:
    template<typename GetFn>
    static AutoControl load_auto(GetFn get_fn) {
        AutoControl control;
        auto res = get_fn();
        if (res.is_error()) {
            return control;
        }
        control.enabled = res.unwrap();
        control.available = true;
        return control;
    }

    template<typename RangeFn, typename GetFn>
    static SliderControl load_slider(RangeFn range_fn, GetFn get_fn) {
        SliderControl control;
        auto range_res = range_fn();
        auto value_res = get_fn();
        if (range_res.is_error() || value_res.is_error()) {
            return control;
        }
        control.range = range_res.unwrap();
        control.value = value_res.unwrap();
        control.available = control.range.min() < control.range.max();
        return control;
    }

    template<typename SetFn>
    void auto_checkbox(const char* label, AutoControl& control, SetFn set_fn) {
        if (!control.available) {
            ImGui::TextDisabled("%s: not supported", label);
            return;
        }
        if (ImGui::Checkbox(label, &control.enabled)) {
            report(label, set_fn(control.enabled));
            // Switching modes can move the manual values (e.g. the focus
            // distance chosen by the autofocus); re-read everything.
            refresh();
        }
    }

    template<typename SetFn>
    void slider(const char* label, SliderControl& control, SetFn set_fn) {
        if (!control.available) {
            ImGui::TextDisabled("%s: not supported", label);
            return;
        }
        if (ImGui::SliderInt(label, &control.value, control.range.min(), control.range.max())) {
            report(label, set_fn(control.value));
        }
    }

    void reset_to_defaults() {
        const auto set_default = [&](SliderControl& control, auto set_fn, const char* label) {
            if (!control.available) {
                return;
            }
            control.value = control.range.default_value();
            report(label, set_fn(control.value));
        };

        set_default(
            focus_,
            [&](int value) { return capture_.set_focus_distance(value); },
            "Focus distance"
        );
        set_default(exposure_, [&](int value) { return capture_.set_exposure(value); }, "Exposure");
        set_default(gain_, [&](int value) { return capture_.set_gain(value); }, "Gain");
        set_default(
            white_balance_,
            [&](int value) { return capture_.set_white_balance_temperature(value); },
            "Temperature"
        );
        set_default(
            brightness_,
            [&](int value) { return capture_.set_brightness(value); },
            "Brightness"
        );
        set_default(contrast_, [&](int value) { return capture_.set_contrast(value); }, "Contrast");
        set_default(
            saturation_,
            [&](int value) { return capture_.set_saturation(value); },
            "Saturation"
        );
        set_default(zoom_, [&](int value) { return capture_.set_zoom(value); }, "Zoom");
    }

    void report(const char* label, const p10::P10Error& error) {
        if (error.is_error()) {
            last_error_ = std::string(label) + ": " + error.to_string();
        } else {
            last_error_.clear();
        }
    }

    MediaCapture capture_;

    AutoControl auto_focus_;
    SliderControl focus_;
    AutoControl auto_exposure_;
    SliderControl exposure_;
    AutoControl auto_white_balance_;
    SliderControl white_balance_;
    SliderControl brightness_;
    SliderControl contrast_;
    SliderControl saturation_;
    SliderControl gain_;
    SliderControl zoom_;

    std::string last_error_;
};

class CameraApp: public p10::viz::GuiApp {
  public:
    explicit CameraApp(const MediaCapture& capture) :
        // Both the player and the panel share the same capture handle, so the
        // controls act on the device that is being displayed.
        video_player_ {capture, *this},
        controls_ {capture} {}

  protected:
    void on_initialize() override {
        video_player_.on_initialize();
        controls_.initialize();
    }

    void on_render() override {
        video_player_.on_render();
        controls_.render();
    }

    void on_cleanup() override {}

  private:
    p10::viz::VideoPlayerComponent video_player_;
    CameraControlsPanel controls_;
};

}  // namespace

int main(int argc, char** argv) {
    const auto cli = parse_args(argc, argv);

    const auto devices = MediaCapture::list_video_devices().expect("Unable to list video devices");

    if (cli.list_devices) {
        std::cout << "Video devices:\n";
        for (const auto& device : devices) {
            std::cout << "  [" << device.index() << "] " << device.name() << '\n';
        }
        return 0;
    }

    if (devices.empty()) {
        std::cerr << "No video capture devices found\n";
        return 1;
    }
    if (cli.device_index >= static_cast<int>(devices.size())) {
        std::cerr << "Invalid device index " << cli.device_index << "; available devices:\n";
        for (const auto& device : devices) {
            std::cerr << "  [" << device.index() << "] " << device.name() << '\n';
        }
        return 1;
    }

    const MediaCapture capture =
        MediaCapture::open_stream(std::make_pair(cli.device_index, p10::media::VideoParameters {}))
            .expect("Unable to open camera");

    CameraApp app(capture);
    const auto params =
        p10::viz::GuiAppParameters().title("camera — " + devices[cli.device_index].name());
    if (auto status = run_app(app, params); status.is_error()) {
        std::cout << status.to_string() << '\n';
    }

    return 0;
}
