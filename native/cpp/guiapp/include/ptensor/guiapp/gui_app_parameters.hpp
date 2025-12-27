#pragma once

#include <string>
namespace p10::guiapp {
class GuiAppParameters {
  public:
    GuiAppParameters& width(int width) {
        width_ = width;
        return *this;
    }

    GuiAppParameters& height(int height) {
        height_ = height;
        return *this;
    }

    GuiAppParameters& title(const std::string& title) {
        title_ = title;
        return *this;
    }

    int width() const {
        return width_;
    }

    int height() const {
        return height_;
    }

    const std::string& title() const {
        return title_;
    }

  private:
    int width_ = 1280;
    int height_ = 720;
    std::string title_ = "p10::guiapp::GuiApp";
};
}