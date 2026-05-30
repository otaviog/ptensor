#pragma once

#include <string>

namespace p10::viz {

/// Builder for GUI application configuration parameters.
class GuiAppParameters {
  public:
    /// Set the window width.
    GuiAppParameters& width(int width) {
        width_ = width;
        return *this;
    }

    /// Set the window height.
    GuiAppParameters& height(int height) {
        height_ = height;
        return *this;
    }

    /// Set the window title.
    GuiAppParameters& title(const std::string& title) {
        title_ = title;
        return *this;
    }

    /// Get the configured window width.
    int width() const {
        return width_;
    }

    /// Get the configured window height.
    int height() const {
        return height_;
    }

    /// Get the configured window title.
    const std::string& title() const {
        return title_;
    }

  private:
    int width_ = 1280;
    int height_ = 720;
    std::string title_ = "p10::viz::GuiApp";
};

}  // namespace p10::viz