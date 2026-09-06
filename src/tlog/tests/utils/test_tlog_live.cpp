// Manual driver for p10::tlog: streams a fixed set of tensors to a running
// tensor log server, so the client, the wire format and the viewer can be
// exercised end to end.
//
//   test_tlog_live --address localhost:8791 --interval 200
//
// The set mirrors `src/ptensor-vscode/cpp/viewer_demo.cpp` (which in turn
// mirrors `src/ptensor-vscode/src/sampleTensors.ts`), so the live path hits the
// same viewer branches the debugger and offline paths do: scalar, vector, 2D
// tables, grayscale/RGB images (planar and interleaved), batched NCHW/NHWC, and
// a large 1D buffer.
//
// Nothing here fails the process: `p10::tlog::log` swallows its errors by
// design, so a server that is down shows up as an error in the log, not as a
// non-zero exit code.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <CLI/CLI.hpp>
#include <ptensor/dtype.hpp>
#include <ptensor/shape.hpp>
#include <ptensor/tensor.hpp>
#include <ptensor/tlog/tlog.hpp>

namespace {

using p10::Dtype;
using p10::make_shape;
using p10::Tensor;

struct TLogLiveCli {
    std::string address;
    int interval_ms = 250;
    int repeat = 1;
    bool list = false;
    std::vector<std::string> only;
};

/// One named tensor of the set. The tensor is built on demand, so `--list` and
/// `--only` cost nothing for the samples they skip.
struct Sample {
    std::string name;
    Tensor (*make)();
};

TLogLiveCli parse_args(int argc, char** argv);
std::vector<Sample> samples();
std::vector<Sample> select(const std::vector<Sample>& all, const std::vector<std::string>& names);
void set_address(const std::string& address);

Tensor scalar_tensor() {
    return Tensor::full(make_shape(1), 3.14159).expect("scalar");
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
    return Tensor::from_range(make_shape(2, 3), p10::TensorOptions(Dtype::Int64), 1)
        .expect("matrix_i64");
}

Tensor grayscale_f32() {
    constexpr int64_t H = 64;
    constexpr int64_t W = 64;
    auto t = Tensor::empty(make_shape(H, W)).expect("gray");
    auto d = t.as_span2d<float>().expect("gray span");
    for (int64_t y = 0; y < H; ++y) {
        for (int64_t x = 0; x < W; ++x) {
            d[y][x] = static_cast<float>(x + y) / static_cast<float>(H + W);
        }
    }
    return t;
}

Tensor rgb_interleaved_u8() {
    constexpr int64_t H = 48;
    constexpr int64_t W = 64;
    auto t = Tensor::empty(make_shape(H, W, 3), p10::TensorOptions(Dtype::Uint8)).expect("rgb_hwc");
    auto d = t.as_accessor3d<uint8_t>().expect("rgb_hwc span");
    for (int64_t y = 0; y < H; ++y) {
        for (int64_t x = 0; x < W; ++x) {
            auto px = d[y][x];
            px[0] = static_cast<uint8_t>(x * 255 / W);
            px[1] = static_cast<uint8_t>(y * 255 / H);
            px[2] = 128;
        }
    }
    return t;
}

Tensor rgb_planar_f32() {
    constexpr int64_t H = 48;
    constexpr int64_t W = 64;
    auto t = Tensor::empty(make_shape(3, H, W)).expect("rgb_chw");
    auto d = t.as_span3d<float>().expect("rgb_chw span");
    // Span3D over a (C, H, W) tensor: channel(c, y) points at plane c, row y.
    for (int64_t y = 0; y < H; ++y) {
        for (int64_t x = 0; x < W; ++x) {
            d[0][y][x] = static_cast<float>(x) / static_cast<float>(W);  // R
            d[1][y][x] = static_cast<float>(y) / static_cast<float>(H);  // G
            d[2][y][x] = 0.5F;  // B
        }
    }
    return t;
}

Tensor batch_nchw_f32() {
    constexpr int64_t N = 2;
    constexpr int64_t H = 32;
    constexpr int64_t W = 32;
    auto t = Tensor::empty(make_shape(N, 3, H, W)).expect("batch_nchw");
    auto d = t.as_span4d<float>().expect("batch_nchw span");
    for (int64_t b = 0; b < N; ++b) {
        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                d[b][0][y][x] =
                    static_cast<float>(x) / static_cast<float>(W) * (b + 1) / static_cast<float>(N);
                d[b][1][y][x] = static_cast<float>(y) / static_cast<float>(H);
                d[b][2][y][x] = static_cast<float>(b) / static_cast<float>(N);
            }
        }
    }
    return t;
}

Tensor batch_nhwc_u8() {
    constexpr int64_t N = 2;
    constexpr int64_t H = 32;
    constexpr int64_t W = 32;
    auto t = Tensor::empty(make_shape(N, H, W, 3), p10::TensorOptions(Dtype::Uint8))
                 .expect("batch_nhwc");
    auto d = t.as_span4d<uint8_t>().expect("batch_nhwc span");
    for (int64_t b = 0; b < N; ++b) {
        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                const auto ramp = static_cast<uint8_t>(x * 255 / W);
                d[b][y][x][0] = (b == 0) ? ramp : static_cast<uint8_t>(255 - ramp);
                d[b][y][x][1] = static_cast<uint8_t>(y * 255 / H);
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

int main(int argc, char** argv) {
    const auto cli = parse_args(argc, argv);
    const auto all = samples();

    if (cli.list) {
        for (const auto& sample : all) {
            std::cout << sample.name << "\n";
        }
        return 0;
    }

    const auto selected = select(all, cli.only);
    if (selected.empty()) {
        std::cerr << "Error: no sample matches --only; run with --list to see the names\n";
        return 1;
    }

    if (!cli.address.empty()) {
        set_address(cli.address);
    }

    const auto interval = std::chrono::milliseconds(cli.interval_ms);
    for (int pass = 0; pass < cli.repeat; ++pass) {
        for (const auto& sample : selected) {
            // Repeats would otherwise send the same name over and over, and the
            // viewer lists one entry per name.
            const std::string entry =
                cli.repeat > 1 ? std::format("{} {}", sample.name, pass) : sample.name;

            p10::tlog::log(entry, sample.make());
            std::cout << "sent " << entry << "\n";

            if (interval.count() > 0) {
                std::this_thread::sleep_for(interval);
            }
        }
    }

    return 0;
}

namespace {

TLogLiveCli parse_args(int argc, char** argv) {
    CLI::App app {
        "Test p10::tlog by sending different tensors.\n\n"
        "Streams the sample tensors of the VS Code viewer demo to a running "
        "tensor log server, one entry per tensor."
    };
    argv = app.ensure_utf8(argv);

    TLogLiveCli cli;

    app.add_option(
        "-a,--address",
        cli.address,
        "Server to log to, as 'host:port'. Sets PTENSOR_TLOG_ADDRESS, which the "
        "client reads on its first log; without either it uses localhost:4449"
    );
    app.add_option("-i,--interval", cli.interval_ms, "Delay between tensors, in ms")
        ->capture_default_str();
    app.add_option("-r,--repeat", cli.repeat, "Number of passes over the samples")
        ->capture_default_str()
        ->check(CLI::PositiveNumber);
    app.add_option("-o,--only", cli.only, "Send only these samples, by name (repeatable)");
    app.add_flag("-l,--list", cli.list, "Print the sample names and exit");

    app.footer(
        "Examples:\n"
        "  test_tlog_live --address localhost:8791\n"
        "  test_tlog_live --only rgb_hwc --only gray --repeat 20 --interval 100\n"
        "  test_tlog_live --list"
    );

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        std::exit(app.exit(e));
    }

    return cli;
}

std::vector<Sample> samples() {
    return {
        {"scalar", scalar_tensor},
        {"vec8", vector_tensor},
        {"mat4x5", matrix_f32},
        {"mat_i64", matrix_i64},
        {"gray", grayscale_f32},
        {"rgb_hwc", rgb_interleaved_u8},
        {"rgb_chw", rgb_planar_f32},
        {"batch_nchw", batch_nchw_f32},
        {"batch_nhwc", batch_nhwc_u8},
        {"large_1d", large_1d_f32},
    };
}

/// The samples named in `names`, in the order they are declared. An empty
/// `names` selects everything.
std::vector<Sample> select(const std::vector<Sample>& all, const std::vector<std::string>& names) {
    if (names.empty()) {
        return all;
    }

    std::vector<Sample> selected;
    for (const auto& sample : all) {
        if (std::ranges::find(names, sample.name) != names.end()) {
            selected.push_back(sample);
        }
    }
    return selected;
}

void set_address(const std::string& address) {
#if defined(_WIN32)
    _putenv_s("PTENSOR_TLOG_ADDRESS", address.c_str());
#else
    setenv("PTENSOR_TLOG_ADDRESS", address.c_str(), 1);
#endif
}

}  // namespace
