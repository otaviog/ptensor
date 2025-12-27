#include <iostream>

#include <ptensor/guiapp/gui_app.hpp>
#include <ptensor/guiapp/gui_app_parameters.hpp>

class WindowApp: public p10::guiapp::GuiApp {
  protected:
    void on_initialize() override {
        // Custom initialization code here
    }

    void on_cleanup() override {
        // Cleanup code here
    }
};

int main(int /*argc*/, char** /*argv*/) {
    WindowApp app;

    if (auto status = app.start(p10::guiapp::GuiAppParameters()); status.is_error()) {
        std::cout << status.to_string() << std::endl;
    }

    return 0;
}