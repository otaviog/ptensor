#pragma once

#include <memory>

#include <imgui.h>
#include <ptensor/p10_error.hpp>

namespace p10::guiapp {
class GuiAppParameters;

class GuiApp {
  private:
    class Impl;

  public:
    GuiApp();

    virtual ~GuiApp();

    P10Error start(const GuiAppParameters& params);

    void quit();

  protected:
    virtual void on_initialize() {}

    virtual void on_render();

    virtual void on_cleanup() {}

  private:
    std::shared_ptr<Impl> impl_;
};
}  // namespace p10::guiapp