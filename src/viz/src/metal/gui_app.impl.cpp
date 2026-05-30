// Metal-CPP private implementations must be defined in exactly ONE translation unit.
// This file owns them for the entire Metal backend.
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "gui_app.impl.hpp"

#include <Metal/Metal.hpp>
#include <SDL.h>
#include <SDL_metal.h>

#include "image_texture.impl.hpp"
#include "imgui_impl_metal.h"
#include "imgui_impl_sdl2.h"

namespace p10::viz {

struct MetalContext {
    MTL::Device* device = nullptr;
    MTL::CommandQueue* command_queue = nullptr;
    CA::MetalLayer* layer = nullptr;  // borrowed from SDL Metal view — not retained here
    SDL_MetalView metal_view = nullptr;
    int width = 0;
    int height = 0;

    ~MetalContext() {
        if (command_queue) {
            command_queue->release();
            command_queue = nullptr;
        }
        if (device) {
            device->release();
            device = nullptr;
        }
    }
};

GuiApp::Impl::Impl(GuiApp& parent) : parent_(parent) {}

GuiApp::Impl::~Impl() {
    if (running_) {
        quit();
    }
}

P10Error GuiApp::Impl::start(const GuiAppParameters& params) {
    params_ = params;

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        return P10Error::InvalidOperation << SDL_GetError();
    }

    window_ = SDL_CreateWindow(
        params.title().c_str(),
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        params.width(),
        params.height(),
        SDL_WINDOW_METAL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI
    );
    if (!window_) {
        SDL_Quit();
        return P10Error::InvalidOperation
            << ("SDL_CreateWindow Error: " + std::string(SDL_GetError()));
    }

    metal_.reset(new MetalContext());

    metal_->metal_view = SDL_Metal_CreateView(window_);
    if (!metal_->metal_view) {
        SDL_DestroyWindow(window_);
        SDL_Quit();
        return P10Error::InvalidOperation << "Failed to create SDL Metal view";
    }

    metal_->device = MTL::CreateSystemDefaultDevice();
    if (!metal_->device) {
        SDL_Metal_DestroyView(metal_->metal_view);
        SDL_DestroyWindow(window_);
        SDL_Quit();
        return P10Error::InvalidOperation << "Failed to create Metal device";
    }

    metal_->layer = reinterpret_cast<CA::MetalLayer*>(SDL_Metal_GetLayer(metal_->metal_view));
    metal_->layer->setDevice(metal_->device);
    metal_->layer->setPixelFormat(MTL::PixelFormatBGRA8Unorm);

    int drawable_w, drawable_h;
    SDL_Metal_GetDrawableSize(window_, &drawable_w, &drawable_h);
    metal_->layer->setDrawableSize(CGSizeMake(drawable_w, drawable_h));
    metal_->width = drawable_w;
    metal_->height = drawable_h;

    metal_->command_queue = metal_->device->newCommandQueue();
    if (!metal_->command_queue) {
        SDL_Metal_DestroyView(metal_->metal_view);
        SDL_DestroyWindow(window_);
        SDL_Quit();
        return P10Error::InvalidOperation << "Failed to create Metal command queue";
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGui_ImplSDL2_InitForMetal(window_);
    ImGui_ImplMetal_Init(metal_->device);

    parent_.on_initialize();

    return P10Error::Ok;
}

void GuiApp::Impl::main_loop() {
    running_ = true;
    while (running_) {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT) {
                running_ = false;
            }
            if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_RESIZED) {
                int w, h;
                SDL_Metal_GetDrawableSize(window_, &w, &h);
                metal_->layer->setDrawableSize(CGSizeMake(w, h));
                metal_->width = w;
                metal_->height = h;
            }
        }

        if (running_) {
            CA::MetalDrawable* drawable = metal_->layer->nextDrawable();
            if (drawable) {
                MTL::RenderPassDescriptor* rpd = MTL::RenderPassDescriptor::renderPassDescriptor();
                MTL::RenderPassColorAttachmentDescriptor* cd = rpd->colorAttachments()->object(0);
                cd->setTexture(drawable->texture());
                cd->setLoadAction(MTL::LoadActionClear);
                cd->setClearColor(MTL::ClearColor::Make(0.0, 0.0, 0.0, 1.0));
                cd->setStoreAction(MTL::StoreActionStore);

                ImGui_ImplMetal_NewFrame(rpd);
                ImGui_ImplSDL2_NewFrame();
                ImGui::NewFrame();

                parent_.on_render();

                ImGui::Render();

                MTL::CommandBuffer* cmd = metal_->command_queue->commandBuffer();
                MTL::RenderCommandEncoder* encoder = cmd->renderCommandEncoder(rpd);
                ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), cmd, encoder);
                encoder->endEncoding();
                cmd->presentDrawable(drawable);
                cmd->commit();
            }
        }

        pool->release();
    }
}

ImageTexture GuiApp::Impl::create_texture() {
    ImageTextureMetalContext ctx;
    ctx.device = metal_->device;
    ctx.command_queue = metal_->command_queue;
    return ImageTexture(new ImageTexture::Impl(ctx));
}

void GuiApp::Impl::quit() {
    running_ = false;
    parent_.on_cleanup();
    ImGui_ImplMetal_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    if (metal_) {
        if (metal_->metal_view) {
            SDL_Metal_DestroyView(metal_->metal_view);
            metal_->metal_view = nullptr;
        }
        metal_.reset();
    }
    if (window_) {
        SDL_DestroyWindow(window_);
        window_ = nullptr;
    }
    SDL_Quit();
}

}  // namespace p10::viz
