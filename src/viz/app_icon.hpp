#pragma once

struct SDL_Window;

namespace p10::viz {

/// Sets the embedded ptensor icon on an SDL window. Best-effort: platforms that
/// ignore SDL window icons (e.g. the macOS dock, which uses the .app bundle)
/// are unaffected. Safe to call with a null window.
void set_app_icon(SDL_Window* window);

}  // namespace p10::viz
