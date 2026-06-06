// Manual test driver for the ptensor VS Code tensor viewer.
//
// Build this target (see CMakeLists.txt in this folder), launch it under the
// LLDB debugger from VS Code, and when it self-traps at `debug_break()` every
// tensor below is in scope. Use the "ptensor: View Tensor" command (or the
// context-menu entry on a variable) to visualize any of them. Internally the
// extension evaluates `p10::to_json_debug(<expr>)`, so anything reachable from
// the stopped frame works.
//
// The set of tensors mirrors `src/ptensor-vscode/src/sampleTensors.ts` so the
// live path exercises the same panel branches the offline samples do: scalar,
// vector, 2D tables, grayscale/RGB images (planar and interleaved), batched
// NCHW/NHWC, and a large 1D buffer.

#include <cmath>
#include <cstdint>
#include <cstdio>

#include <ptensor/dtype.hpp>
#include <ptensor/shape.hpp>
#include <ptensor/tensor.hpp>
#include <ptensor/tensor_print.hpp>

namespace {

using p10::Dtype;
using p10::make_shape;
using p10::Tensor;

// Stops the process so the debugger can inspect the frame. Under LLDB this
// raises a trap the debugger catches; outside a debugger it is typically a
// no-op crash, which is fine for this manual-only driver.
void debug_break() {
#if defined(__clang__) || defined(__GNUC__)
    __builtin_debugtrap();
#endif
}

Tensor scalar_tensor() {
    auto t = Tensor::full(make_shape(1), 3.14159).expect("scalar");
    return t;
}

Tensor vector_tensor() {
    return Tensor::from_range(make_shape(8)).expect("vector");
}

Tensor matrix_f32() {
    auto t = Tensor::from_range(make_shape(4, 5)).expect("matrix_f32");
    auto data = t.as_span1d<float>().expect("matrix_f32 span");
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i) - 7.5F;
    }
    return t;
}

Tensor matrix_i64() {
    auto t =
        Tensor::from_range(make_shape(2, 3), p10::TensorOptions(Dtype::Int64), 1).expect("matrix_i64"
        );
    return t;
}

Tensor grayscale_f32() {
    constexpr int64_t h = 64;
    constexpr int64_t w = 64;
    auto t = Tensor::empty(make_shape(h, w)).expect("gray");
    auto d = t.as_span2d<float>().expect("gray span");
    for (int64_t y = 0; y < h; ++y) {
        for (int64_t x = 0; x < w; ++x) {
            d[y][x] = static_cast<float>(x + y) / static_cast<float>(h + w);
        }
    }
    return t;
}

Tensor rgb_interleaved_u8() {
    constexpr int64_t h = 48;
    constexpr int64_t w = 64;
    auto t = Tensor::empty(make_shape(h, w, 3), p10::TensorOptions(Dtype::Uint8)).expect("rgb_hwc");
    auto d = t.as_span3d<uint8_t>().expect("rgb_hwc span");
    for (int64_t y = 0; y < h; ++y) {
        for (int64_t x = 0; x < w; ++x) {
            auto* px = d.channel(y, x);
            px[0] = static_cast<uint8_t>(x * 255 / w);
            px[1] = static_cast<uint8_t>(y * 255 / h);
            px[2] = 128;
        }
    }
    return t;
}

Tensor rgb_planar_f32() {
    constexpr int64_t h = 48;
    constexpr int64_t w = 64;
    auto t = Tensor::empty(make_shape(3, h, w)).expect("rgb_chw");
    auto d = t.as_planar_span3d<float>().expect("rgb_chw span");
    // Span3D over a (C, H, W) tensor: channel(c, y) points at plane c, row y.
    for (int64_t y = 0; y < h; ++y) {
        for (int64_t x = 0; x < w; ++x) {
            d[0][y][x] = static_cast<float>(x) / static_cast<float>(w);  // R
            d[1][y][x] = static_cast<float>(y) / static_cast<float>(h);  // G
            d[2][y][x] = 0.5F;                                           // B
        }
    }
    return t;
}

Tensor batch_nchw_f32() {
    constexpr int64_t n = 2;
    constexpr int64_t h = 32;
    constexpr int64_t w = 32;
    auto t = Tensor::empty(make_shape(n, 3, h, w)).expect("batch_nchw");
    auto d = t.as_span4d<float>().expect("batch_nchw span");
    for (int64_t b = 0; b < n; ++b) {
        for (int64_t y = 0; y < h; ++y) {
            for (int64_t x = 0; x < w; ++x) {
                d[b][0][y][x] =
                    static_cast<float>(x) / static_cast<float>(w) * (b + 1) / static_cast<float>(n);
                d[b][1][y][x] = static_cast<float>(y) / static_cast<float>(h);
                d[b][2][y][x] = static_cast<float>(b) / static_cast<float>(n);
            }
        }
    }
    return t;
}

Tensor batch_nhwc_u8() {
    constexpr int64_t n = 2;
    constexpr int64_t h = 32;
    constexpr int64_t w = 32;
    auto t =
        Tensor::empty(make_shape(n, h, w, 3), p10::TensorOptions(Dtype::Uint8)).expect("batch_nhwc");
    auto d = t.as_span4d<uint8_t>().expect("batch_nhwc span");
    for (int64_t b = 0; b < n; ++b) {
        for (int64_t y = 0; y < h; ++y) {
            for (int64_t x = 0; x < w; ++x) {
                const auto ramp = static_cast<uint8_t>(x * 255 / w);
                d[b][y][x][0] = (b == 0) ? ramp : static_cast<uint8_t>(255 - ramp);
                d[b][y][x][1] = static_cast<uint8_t>(y * 255 / h);
                d[b][y][x][2] = (b == 0) ? 64 : 192;
            }
        }
    }
    return t;
}

Tensor large_1d_f32() {
    auto t = Tensor::empty(make_shape(1024)).expect("large_1d");
    auto d = t.as_span1d<float>().expect("large_1d span");
    for (size_t i = 0; i < d.size(); ++i) {
        d[i] = std::sin(static_cast<float>(i) / 16.0F);
    }
    return t;
}

}  // namespace

int main() {
    // Each of these is in scope at the breakpoint below. View any of them with
    // the "ptensor: View Tensor" command.
    Tensor scalar = scalar_tensor();
    Tensor vec8 = vector_tensor();
    Tensor mat4x5 = matrix_f32();
    Tensor mat_i64 = matrix_i64();
    Tensor gray = grayscale_f32();
    Tensor rgb_hwc = rgb_interleaved_u8();
    Tensor rgb_chw = rgb_planar_f32();
    Tensor batch_nchw = batch_nchw_f32();
    Tensor batch_nhwc = batch_nhwc_u8();
    Tensor large_1d = large_1d_f32();

    std::printf("ptensor viewer demo: %d tensors ready.\n", 11);
    // Using the result of to_json_debug forces the linker to pull tensor_print.o
    // out of the static ptensor lib, so the symbol exists in the binary and the
    // debugger's expression evaluator can call p10::to_json_debug(<tensor>).
    std::printf("scalar as json: %s\n", p10::to_json_debug(scalar));
    std::printf("Stopping for the debugger; view a tensor with 'ptensor: View Tensor'.\n");

    // >>> Debugger stops here. All tensors above are live in this frame. <<<
    debug_break();

    // Keep everything referenced so nothing is optimized out before the trap.
    return static_cast<int>(
        scalar.size() + vec8.size() + mat4x5.size() + mat_i64.size() + gray.size()
        + rgb_hwc.size() + rgb_chw.size() + batch_nchw.size() + batch_nhwc.size() + large_1d.size()
        & 0
    );
}
