#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <ptensor/shape.hpp>
#include <ptensor/tensor.hpp>
#include <ptensor/tensor_options.hpp>
#include <ptensor/testing/catch2_assertions.hpp>

namespace p10 {

// ============================================================================
// Tensor Creation Tests
// ============================================================================

TEST_CASE("Tensor::full creates tensor filled with specified value", "[tensor][creation]") {
    auto tensor = Tensor::full(make_shape({2, 3}).unwrap(), 3.0).expect("Could not create tensor");

    REQUIRE(tensor.dtype() == Dtype::Float32);
    REQUIRE(tensor.shape().count() == 6);
    REQUIRE(tensor.stride().dims() == 2);
    REQUIRE(tensor.axes().dims() == 2);
    REQUIRE(tensor.dims() == 2);

    auto data = tensor.as_span1d<float>().unwrap();
    for (auto i = 0; i < tensor.shape().count(); i++) {
        REQUIRE(data[i] == Catch::Approx(3.0f));
    }
}

TEST_CASE("Tensor::full with different data types", "[tensor][creation]") {
    SECTION("int32") {
        auto tensor =
            Tensor::full(make_shape({3, 3}).unwrap(), 42, TensorOptions().dtype(Dtype::Int32))
                .unwrap();
        REQUIRE(tensor.dtype() == Dtype::Int32);
        auto data = tensor.as_span1d<int32_t>().unwrap();
        for (auto i = 0; i < 9; i++) {
            REQUIRE(data[i] == 42);
        }
    }

    SECTION("float64") {
        auto tensor =
            Tensor::full(make_shape({2, 2}).unwrap(), 3.14, TensorOptions().dtype(Dtype::Float64))
                .unwrap();
        REQUIRE(tensor.dtype() == Dtype::Float64);
        auto data = tensor.as_span1d<double>().unwrap();
        for (auto i = 0; i < 4; i++) {
            REQUIRE(data[i] == Catch::Approx(3.14));
        }
    }
}

TEST_CASE("Tensor::zeros creates tensor filled with zeros", "[tensor][creation]") {
    auto tensor = Tensor::zeros(make_shape({2, 3}).unwrap()).expect("Could not create tensor");

    REQUIRE(tensor.dtype() == Dtype::Float32);
    REQUIRE(tensor.shape().count() == 6);
    REQUIRE(tensor.stride().dims() == 2);
    REQUIRE(tensor.axes().dims() == 2);
    REQUIRE(tensor.dims() == 2);

    auto data = tensor.as_span1d<float>().unwrap();
    for (auto i = 0; i < tensor.shape().count(); i++) {
        REQUIRE(data[i] == Catch::Approx(0.0f));
    }
}

TEST_CASE("Tensor::zeros with explicit dtype", "[tensor][creation]") {
    auto tensor =
        Tensor::zeros(make_shape({3, 4}).unwrap(), TensorOptions().dtype(Dtype::Int64)).unwrap();

    REQUIRE(tensor.dtype() == Dtype::Int64);
    auto data = tensor.as_span1d<int64_t>().unwrap();
    for (size_t i = 0; i < 12; i++) {
        REQUIRE(data[i] == 0);
    }
}

TEST_CASE("Tensor::empty allocates uninitialized tensor", "[tensor][creation]") {
    auto tensor = Tensor::empty(make_shape({2, 3}).unwrap()).expect("Could not create tensor");

    REQUIRE(tensor.dtype() == Dtype::Float32);
    REQUIRE(tensor.shape().count() == 6);
    REQUIRE(tensor.stride().dims() == 2);
    REQUIRE(tensor.axes().dims() == 2);
    REQUIRE(tensor.dims() == 2);
}

TEST_CASE("Tensor::empty with various shapes", "[tensor][creation]") {
    SECTION("1D tensor") {
        auto tensor = Tensor::empty(make_shape({10}).unwrap()).unwrap();
        REQUIRE(tensor.dims() == 1);
        REQUIRE(tensor.shape().count() == 10);
    }

    SECTION("3D tensor") {
        auto tensor = Tensor::empty(make_shape({2, 3, 4}).unwrap()).unwrap();
        REQUIRE(tensor.dims() == 3);
        REQUIRE(tensor.shape().count() == 24);
    }

    SECTION("4D tensor") {
        auto tensor = Tensor::empty(make_shape({2, 3, 4, 5}).unwrap()).unwrap();
        REQUIRE(tensor.dims() == 4);
        REQUIRE(tensor.shape().count() == 120);
    }
}

// ============================================================================
// Tensor Properties Tests
// ============================================================================

TEST_CASE("Tensor dtype and device can be configured", "[tensor][properties]") {
    const auto all_dtypes = {
        Dtype::Uint8,
        Dtype::Uint16,
        Dtype::Uint32,
        Dtype::Int8,
        Dtype::Int16,
        Dtype::Int32,
        Dtype::Int64,
        Dtype::Float16,
        Dtype::Float32,
        Dtype::Float64
    };

    const auto all_devices = {Device::Cpu, Device::Cuda};

    for (const auto& dtype : all_dtypes) {
        for (const auto& device : all_devices) {
            auto tensor = Tensor::from_data(
                nullptr,
                make_shape({2, 3}).unwrap(),
                TensorOptions().dtype(dtype).device(device)
            );

            REQUIRE(tensor.dtype() == dtype);
            REQUIRE(tensor.device() == device);
        }
    }
}

TEST_CASE("Tensor strides are computed correctly", "[tensor][properties]") {
    SECTION("default row-major strides") {
        auto tensor = Tensor::empty(make_shape({2, 3}).unwrap()).expect("Could not create tensor");

        REQUIRE(tensor.stride().dims() == 2);
        REQUIRE(tensor.stride(0).unwrap() == 3);
        REQUIRE(tensor.stride(1).unwrap() == 1);
    }

    SECTION("custom strides") {
        auto tensor = Tensor::empty(
                          make_shape({2, 3}).unwrap(),
                          TensorOptions().stride(make_stride({1, 2}).unwrap())
        )
                          .expect("Could not create tensor");

        REQUIRE(tensor.stride().dims() == 2);
        REQUIRE(tensor.stride(0).unwrap() == 1);
        REQUIRE(tensor.stride(1).unwrap() == 2);
    }

    SECTION("3D strides") {
        auto tensor = Tensor::empty(make_shape({2, 3, 4}).unwrap()).unwrap();
        REQUIRE(tensor.stride(0).unwrap() == 12);
        REQUIRE(tensor.stride(1).unwrap() == 4);
        REQUIRE(tensor.stride(2).unwrap() == 1);
    }
}

TEST_CASE("Tensor dimensions are tracked correctly", "[tensor][properties]") {
    auto tensor =
        Tensor::empty(make_shape({2, 3, 4, 5}).unwrap()).expect("Unable to create tensor");

    REQUIRE(tensor.dims() == 4);
    REQUIRE(tensor.shape().dims() == 4);
    REQUIRE(tensor.stride().dims() == 4);
    REQUIRE(tensor.axes().dims() == 4);
}

TEST_CASE("Tensor::empty() detects empty tensors", "[tensor][properties]") {
    SECTION("non-empty tensor") {
        auto tensor = Tensor::empty(make_shape({2, 3}).unwrap()).unwrap();
        REQUIRE_FALSE(tensor.empty());
    }

    SECTION("tensor with zero dimension") {
        auto tensor = Tensor::empty(make_shape({0, 3}).unwrap()).unwrap();
        REQUIRE(tensor.empty());
    }

    SECTION("tensor with multiple zero dimensions") {
        auto tensor = Tensor::empty(make_shape({0, 0}).unwrap()).unwrap();
        REQUIRE(tensor.empty());
    }
}

// ============================================================================
// Tensor Contiguity Tests
// ============================================================================

TEST_CASE("Tensor::is_contiguous detects memory layout", "[tensor][contiguity]") {
    SECTION("default tensors are contiguous") {
        auto tensor = Tensor::empty(make_shape({2, 3}).unwrap()).unwrap();
        REQUIRE(tensor.is_contiguous());
    }

    SECTION("custom strides may not be contiguous") {
        auto tensor = Tensor::empty(
                          make_shape({2, 3}).unwrap(),
                          TensorOptions().stride(make_stride({1, 2}).unwrap())
        )
                          .unwrap();
        REQUIRE_FALSE(tensor.is_contiguous());
    }
}

TEST_CASE("Tensor::to_contiguous creates contiguous copy", "[tensor][contiguity]") {
    auto tensor = Tensor::empty(
                      make_shape({2, 3}).unwrap(),
                      TensorOptions().stride(make_stride({1, 2}).unwrap())
    )
                      .unwrap();

    // Original data layout:
    // Physical: 0 1 2 3 4 5
    // Logical view (stride {1, 2}):
    // Row 0: 0 2 4
    // Row 1: 1 3 5
    for (auto i = 0; i < tensor.shape().count(); i++) {
        tensor.as_span1d<float>().unwrap()[i] = float(i);
    }

    auto contiguous = tensor.to_contiguous().unwrap();
    REQUIRE(contiguous.is_contiguous());

    // Contiguous layout should preserve logical view:
    // 0 2 4 1 3 5
    const std::array<float, 6> expected = {0, 2, 4, 1, 3, 5};
    for (auto i = 0; i < contiguous.shape().count(); i++) {
        REQUIRE(contiguous.as_span1d<float>().unwrap()[i] == Catch::Approx(expected[i]));
    }
}

TEST_CASE("Tensor::to_contiguous on already contiguous tensor", "[tensor][contiguity]") {
    auto tensor = Tensor::full(make_shape({3, 3}).unwrap(), 1.0).unwrap();
    REQUIRE(tensor.is_contiguous());

    auto contiguous = tensor.to_contiguous().unwrap();
    REQUIRE(contiguous.is_contiguous());

    REQUIRE_THAT(testing::compare_tensors(tensor, contiguous), testing::IsOk());
}

// ============================================================================
// Tensor View Tests
// ============================================================================

TEST_CASE("Tensor::as_view creates view sharing data", "[tensor][view]") {
    auto tensor = Tensor::zeros(make_shape({2, 3}).unwrap()).unwrap();

    auto tensor_view = tensor.as_view();

    SECTION("view has same properties") {
        REQUIRE(tensor_view.shape().dims() == 2);
        REQUIRE(tensor_view.stride().dims() == 2);
        REQUIRE(tensor_view.axes().dims() == 2);
        REQUIRE(tensor_view.dims() == 2);
    }

    SECTION("view shares underlying data") {
        tensor_view.as_span1d<float>().unwrap()[0] = 3.0;
        REQUIRE(tensor.as_span1d<float>().unwrap()[0] == Catch::Approx(3.0f));

        tensor.as_span1d<float>().unwrap()[1] = 5.0;
        REQUIRE(tensor_view.as_span1d<float>().unwrap()[1] == Catch::Approx(5.0f));
    }
}

TEST_CASE("Tensor::as_view with non-contiguous tensor", "[tensor][view]") {
    auto tensor = Tensor::full(
                      make_shape({3, 4}).unwrap(),
                      2.5,
                      TensorOptions().stride(make_stride({2, 1}).unwrap())
    )
                      .unwrap();

    auto view = tensor.as_view();
    REQUIRE(view.stride() == tensor.stride());
    REQUIRE_FALSE(view.is_contiguous());
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_CASE("Tensor handles single element", "[tensor][edge-cases]") {
    auto tensor = Tensor::full(make_shape({1}).unwrap(), 42.0).unwrap();
    REQUIRE(tensor.size() == 1);
    REQUIRE(tensor.as_span1d<float>().unwrap()[0] == Catch::Approx(42.0f));
}

TEST_CASE("Tensor with large dimensions", "[tensor][edge-cases]") {
    auto tensor = Tensor::empty(make_shape({100, 200}).unwrap()).unwrap();
    REQUIRE(tensor.size() == 20000);
    REQUIRE(tensor.is_contiguous());
}

// ============================================================================
// Tensor Span Access Tests
// ============================================================================

TEST_CASE("Tensor::as_bytes provides byte-level access", "[tensor][span]") {
    auto tensor = Tensor::full(make_shape({2, 3}).unwrap(), 42.0f).unwrap();

    SECTION("const version") {
        const auto& const_tensor = tensor;
        auto bytes = const_tensor.as_bytes();
        REQUIRE(bytes.size() == tensor.size_bytes());
        REQUIRE(bytes.size() == 6 * sizeof(float));
    }

    SECTION("mutable version") {
        auto bytes = tensor.as_bytes();
        REQUIRE(bytes.size() == tensor.size_bytes());
        REQUIRE(bytes.size() == 6 * sizeof(float));
    }
}

TEST_CASE("Tensor::as_span1d converts to 1D span", "[tensor][span]") {
    SECTION("creates span for float tensor") {
        auto tensor = Tensor::full(make_shape({2, 3}).unwrap(), 1.5f).unwrap();
        auto span = tensor.as_span1d<float>().unwrap();

        REQUIRE(span.size() == 6);
        for (size_t i = 0; i < span.size(); i++) {
            REQUIRE(span[i] == Catch::Approx(1.5f));
        }
    }

    SECTION("const version works correctly") {
        auto tensor = Tensor::full(make_shape({4}).unwrap(), 2.5f).unwrap();
        const auto& const_tensor = tensor;
        auto span = const_tensor.as_span1d<float>().unwrap();

        REQUIRE(span.size() == 4);
        REQUIRE(span[0] == Catch::Approx(2.5f));
    }

    SECTION("supports int32 dtype") {
        auto tensor = Tensor::full(make_shape({5}).unwrap(), 10, TensorOptions().dtype(Dtype::Int32))
                          .unwrap();
        auto span = tensor.as_span1d<int32_t>().unwrap();

        REQUIRE(span.size() == 5);
        for (size_t i = 0; i < span.size(); i++) {
            REQUIRE(span[i] == 10);
        }
    }

    SECTION("supports double dtype") {
        auto tensor =
            Tensor::full(make_shape({3}).unwrap(), 3.14, TensorOptions().dtype(Dtype::Float64))
                .unwrap();
        auto span = tensor.as_span1d<double>().unwrap();

        REQUIRE(span.size() == 3);
        for (size_t i = 0; i < span.size(); i++) {
            REQUIRE(span[i] == Catch::Approx(3.14));
        }
    }

    SECTION("fails with wrong dtype") {
        auto tensor = Tensor::full(make_shape({3}).unwrap(), 1.0f).unwrap();
        auto result = tensor.as_span1d<int32_t>();

        REQUIRE(result.is_error());
        REQUIRE(result.err().code() == PtensorError::InvalidArgument);
    }

    SECTION("mutable span allows modification") {
        auto tensor = Tensor::zeros(make_shape({4}).unwrap()).unwrap();
        auto span = tensor.as_span1d<float>().unwrap();

        span[0] = 1.0f;
        span[1] = 2.0f;
        span[2] = 3.0f;
        span[3] = 4.0f;

        auto verify = tensor.as_span1d<float>().unwrap();
        REQUIRE(verify[0] == Catch::Approx(1.0f));
        REQUIRE(verify[1] == Catch::Approx(2.0f));
        REQUIRE(verify[2] == Catch::Approx(3.0f));
        REQUIRE(verify[3] == Catch::Approx(4.0f));
    }
}

TEST_CASE("Tensor::as_span2d converts to 2D span", "[tensor][span]") {
    SECTION("creates 2D span for float tensor") {
        auto tensor = Tensor::zeros(make_shape({3, 4}).unwrap()).unwrap();
        auto span = tensor.as_span2d<float>().unwrap();

        REQUIRE(span.height() == 3);
        REQUIRE(span.width() == 4);

        // Set values using 2D access
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 4; j++) {
                span.row(i)[j] = float(i * 4 + j);
            }
        }

        // Verify values
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 4; j++) {
                REQUIRE(span.row(i)[j] == Catch::Approx(float(i * 4 + j)));
            }
        }
    }

    SECTION("const version works correctly") {
        auto tensor = Tensor::full(make_shape({2, 5}).unwrap(), 7.0f).unwrap();
        const auto& const_tensor = tensor;
        auto span = const_tensor.as_span2d<float>().unwrap();

        REQUIRE(span.height() == 2);
        REQUIRE(span.width() == 5);
        REQUIRE(span.row(0)[0] == Catch::Approx(7.0f));
        REQUIRE(span.row(1)[4] == Catch::Approx(7.0f));
    }

    SECTION("supports int64 dtype") {
        auto tensor =
            Tensor::full(make_shape({2, 3}).unwrap(), 100, TensorOptions().dtype(Dtype::Int64))
                .unwrap();
        auto span = tensor.as_span2d<int64_t>().unwrap();

        REQUIRE(span.height() == 2);
        REQUIRE(span.width() == 3);
        REQUIRE(span.row(0)[0] == 100);
    }

    SECTION("fails for non-2D tensor") {
        auto tensor = Tensor::zeros(make_shape({2, 3, 4}).unwrap()).unwrap();
        auto result = tensor.as_span2d<float>();

        REQUIRE(result.is_error());
        REQUIRE(result.err().code() == PtensorError::InvalidArgument);
    }

    SECTION("fails with wrong dtype") {
        auto tensor = Tensor::zeros(make_shape({2, 3}).unwrap()).unwrap();
        auto result = tensor.as_span2d<double>();

        REQUIRE(result.is_error());
        REQUIRE(result.err().code() == PtensorError::InvalidArgument);
    }
}

TEST_CASE("Tensor::as_span3d converts to 3D span", "[tensor][span]") {
    SECTION("creates 3D span for float tensor") {
        auto tensor = Tensor::zeros(make_shape({2, 3, 4}).unwrap()).unwrap();
        auto span = tensor.as_span3d<float>().unwrap();

        REQUIRE(span.height() == 2);
        REQUIRE(span.width() == 3);
        REQUIRE(span.channels() == 4);

        // Set values using 3D access
        for (size_t h = 0; h < 2; h++) {
            for (size_t w = 0; w < 3; w++) {
                for (size_t c = 0; c < 4; c++) {
                    span.channel(h, w)[c] = float(h * 12 + w * 4 + c);
                }
            }
        }

        // Verify values
        REQUIRE(span.channel(0, 0)[0] == Catch::Approx(0.0f));
        REQUIRE(span.channel(0, 0)[3] == Catch::Approx(3.0f));
        REQUIRE(span.channel(0, 2)[2] == Catch::Approx(10.0f));
        REQUIRE(span.channel(1, 1)[1] == Catch::Approx(17.0f));
    }

    SECTION("const version works correctly") {
        auto tensor = Tensor::full(make_shape({2, 2, 3}).unwrap(), 5.5f).unwrap();
        const auto& const_tensor = tensor;
        auto span = const_tensor.as_span3d<float>().unwrap();

        REQUIRE(span.height() == 2);
        REQUIRE(span.width() == 2);
        REQUIRE(span.channels() == 3);
        REQUIRE(span.channel(0, 0)[0] == Catch::Approx(5.5f));
        REQUIRE(span.channel(1, 1)[2] == Catch::Approx(5.5f));
    }

    SECTION("supports uint8 dtype") {
        auto tensor =
            Tensor::full(make_shape({2, 2, 3}).unwrap(), 128, TensorOptions().dtype(Dtype::Uint8))
                .unwrap();
        auto span = tensor.as_span3d<uint8_t>().unwrap();

        REQUIRE(span.height() == 2);
        REQUIRE(span.width() == 2);
        REQUIRE(span.channels() == 3);
        REQUIRE(span.channel(0, 0)[0] == 128);
    }

    SECTION("row access works correctly") {
        auto tensor = Tensor::zeros(make_shape({3, 4, 2}).unwrap()).unwrap();
        auto span = tensor.as_span3d<float>().unwrap();

        auto row0 = span.row(0);
        auto row1 = span.row(1);

        // Each row should point to width * channels elements
        row0[0] = 1.0f;
        row1[0] = 2.0f;

        REQUIRE(span.channel(0, 0)[0] == Catch::Approx(1.0f));
        REQUIRE(span.channel(1, 0)[0] == Catch::Approx(2.0f));
    }

    SECTION("fails for non-3D tensor") {
        auto tensor = Tensor::zeros(make_shape({2, 3}).unwrap()).unwrap();
        auto result = tensor.as_span3d<float>();

        REQUIRE(result.is_error());
        REQUIRE(result.err().code() == PtensorError::InvalidArgument);
    }

    SECTION("fails with wrong dtype") {
        auto tensor = Tensor::zeros(make_shape({2, 3, 4}).unwrap()).unwrap();
        auto result = tensor.as_span3d<int32_t>();

        REQUIRE(result.is_error());
        REQUIRE(result.err().code() == PtensorError::InvalidArgument);
    }
}

TEST_CASE("Tensor::as_planar_span3d converts to planar 3D span", "[tensor][span]") {
    SECTION("creates planar 3D span for float tensor") {
        auto tensor = Tensor::zeros(make_shape({3, 4, 5}).unwrap()).unwrap();
        auto span = tensor.as_planar_span3d<float>().unwrap();

        REQUIRE(span.channels() == 3);
        REQUIRE(span.height() == 4);
        REQUIRE(span.width() == 5);

        // Set values in each plane
        for (size_t c = 0; c < 3; c++) {
            auto plane = span.plane(c);
            for (size_t h = 0; h < 4; h++) {
                for (size_t w = 0; w < 5; w++) {
                    plane.row(h)[w] = float(c * 100 + h * 10 + w);
                }
            }
        }

        // Verify values
        REQUIRE(span.plane(0).row(0)[0] == Catch::Approx(0.0f));
        REQUIRE(span.plane(0).row(1)[2] == Catch::Approx(12.0f));
        REQUIRE(span.plane(1).row(2)[3] == Catch::Approx(123.0f));
        REQUIRE(span.plane(2).row(3)[4] == Catch::Approx(234.0f));
    }

    SECTION("const version works correctly") {
        auto tensor = Tensor::full(make_shape({2, 3, 4}).unwrap(), 9.0f).unwrap();
        const auto& const_tensor = tensor;
        auto span = const_tensor.as_planar_span3d<float>().unwrap();

        REQUIRE(span.channels() == 2);
        REQUIRE(span.height() == 3);
        REQUIRE(span.width() == 4);

        auto plane0 = span.plane(0);
        REQUIRE(plane0.row(0)[0] == Catch::Approx(9.0f));
        REQUIRE(plane0.row(2)[3] == Catch::Approx(9.0f));
    }

    SECTION("supports uint16 dtype") {
        auto tensor =
            Tensor::full(make_shape({2, 3, 4}).unwrap(), 1000, TensorOptions().dtype(Dtype::Uint16))
                .unwrap();
        auto span = tensor.as_planar_span3d<uint16_t>().unwrap();

        REQUIRE(span.channels() == 2);
        REQUIRE(span.height() == 3);
        REQUIRE(span.width() == 4);
        REQUIRE(span.plane(0).row(0)[0] == 1000);
    }

    SECTION("plane dimensions are correct") {
        auto tensor = Tensor::zeros(make_shape({3, 10, 20}).unwrap()).unwrap();
        auto span = tensor.as_planar_span3d<float>().unwrap();

        for (size_t c = 0; c < 3; c++) {
            auto plane = span.plane(c);
            REQUIRE(plane.height() == 10);
            REQUIRE(plane.width() == 20);
        }
    }

    SECTION("fails for non-3D tensor") {
        auto tensor = Tensor::zeros(make_shape({2, 3, 4, 5}).unwrap()).unwrap();
        auto result = tensor.as_planar_span3d<float>();

        REQUIRE(result.is_error());
        REQUIRE(result.err().code() == PtensorError::InvalidArgument);
    }

    SECTION("fails with wrong dtype") {
        auto tensor = Tensor::zeros(make_shape({2, 3, 4}).unwrap()).unwrap();
        auto result = tensor.as_planar_span3d<double>();

        REQUIRE(result.is_error());
        REQUIRE(result.err().code() == PtensorError::InvalidArgument);
    }
}

}  // namespace p10
