#include <cstdint>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <ptensor/tensor.hpp>
#include <ptensor/testing/catch2_assertions.hpp>
#include <ptensor/testing/compare_tensors.hpp>

namespace p10 {

// ============================================================================
// Tensor Creation Tests
// ============================================================================

TEST_CASE("Tensor::full creates tensor filled with specified value", "[tensor][creation]") {
    auto tensor = Tensor::full(make_shape(2, 3), 3.0).expect("Could not create tensor");

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
            Tensor::full(make_shape(3, 3), 42, TensorOptions().dtype(Dtype::Int32)).unwrap();
        REQUIRE(tensor.dtype() == Dtype::Int32);
        auto data = tensor.as_span1d<int32_t>().unwrap();
        for (auto i = 0; i < 9; i++) {
            REQUIRE(data[i] == 42);
        }
    }

    SECTION("float64") {
        auto tensor =
            Tensor::full(make_shape(2, 2), 3.14, TensorOptions().dtype(Dtype::Float64)).unwrap();
        REQUIRE(tensor.dtype() == Dtype::Float64);
        auto data = tensor.as_span1d<double>().unwrap();
        for (auto i = 0; i < 4; i++) {
            REQUIRE(data[i] == Catch::Approx(3.14));
        }
    }
}

TEST_CASE("Tensor::zeros creates tensor filled with zeros", "[tensor][creation]") {
    auto tensor = Tensor::zeros(make_shape(2, 3)).expect("Could not create tensor");

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
    auto tensor = Tensor::zeros(make_shape(3, 4), TensorOptions().dtype(Dtype::Int64)).unwrap();

    REQUIRE(tensor.dtype() == Dtype::Int64);
    auto data = tensor.as_span1d<int64_t>().unwrap();
    for (size_t i = 0; i < 12; i++) {
        REQUIRE(data[i] == 0);
    }
}

TEST_CASE("Tensor::empty allocates uninitialized tensor", "[tensor][creation]") {
    auto tensor = Tensor::empty(make_shape(2, 3)).expect("Could not create tensor");

    REQUIRE(tensor.dtype() == Dtype::Float32);
    REQUIRE(tensor.shape().count() == 6);
    REQUIRE(tensor.stride().dims() == 2);
    REQUIRE(tensor.axes().dims() == 2);
    REQUIRE(tensor.dims() == 2);
}

TEST_CASE("Tensor::empty with various shapes", "[tensor][creation]") {
    SECTION("1D tensor") {
        auto tensor = Tensor::empty(make_shape(10)).unwrap();
        REQUIRE(tensor.dims() == 1);
        REQUIRE(tensor.shape().count() == 10);
    }

    SECTION("3D tensor") {
        auto tensor = Tensor::empty(make_shape(2, 3, 4)).unwrap();
        REQUIRE(tensor.dims() == 3);
        REQUIRE(tensor.shape().count() == 24);
    }

    SECTION("4D tensor") {
        auto tensor = Tensor::empty(make_shape(2, 3, 4, 5)).unwrap();
        REQUIRE(tensor.dims() == 4);
        REQUIRE(tensor.shape().count() == 120);
    }
}

// ===========================================================================
// Tensor move constructs
// ===========================================================================

TEST_CASE("Tensor move constructor transfers ownership", "[tensor][move]") {
    auto original = Tensor::full(make_shape(2, 2), 7.0, Dtype::Int32).unwrap();
    auto original_data_ptr = original.as_span1d<int32_t>().unwrap().data();

    Tensor moved(std::move(original));

    REQUIRE(moved.dtype() == Dtype::Int32);
    REQUIRE(moved.shape().count() == 4);
    REQUIRE(moved.as_span1d<int32_t>().unwrap().data() == original_data_ptr);

    // Original tensor should be in a valid but unspecified state
    REQUIRE(original.as_span1d<float>().unwrap().data() == nullptr);
    REQUIRE(original.shape().count() == 0);
    REQUIRE(original.dims() == 0);
    REQUIRE(original.dtype() == Dtype::Float32);
}

TEST_CASE("Tensor move assign transfers ownership", "[tensor][move]") {
    auto original = Tensor::full(make_shape(2, 2), 7.0, Dtype::Int32).unwrap();
    auto original_data_ptr = original.as_span1d<int32_t>().unwrap().data();

    Tensor moved = std::move(original);

    REQUIRE(moved.dtype() == Dtype::Int32);
    REQUIRE(moved.shape().count() == 4);
    REQUIRE(moved.as_span1d<int32_t>().unwrap().data() == original_data_ptr);

    // Original tensor should be in a valid but unspecified state
    REQUIRE(original.as_span1d<float>().unwrap().data() == nullptr);
    REQUIRE(original.shape().count() == 0);
    REQUIRE(original.dims() == 0);
    REQUIRE(original.dtype() == Dtype::Float32);
}

// ============================================================================
// Tensor Copy
// ============================================================================

TEST_CASE("Tensor::copy_from copies data from another tensor", "[tensor][copy]") {
    SECTION("Empty dest tensor") {
        auto source = Tensor::full(make_shape(2, 3), 5.0, Dtype::Float32).unwrap();
        Tensor destination;

        REQUIRE(destination.copy_from(source).is_ok());

        auto dest_data = destination.as_span1d<float>().unwrap();
        for (auto i = 0; i < destination.shape().count(); i++) {
            REQUIRE(dest_data[i] == Catch::Approx(5.0f));
        }
    }

    SECTION("Preallocated dest tensor") {
        auto source = Tensor::full(make_shape(2, 3), 8.0, Dtype::Float32).unwrap();
        auto destination =
            Tensor::empty(make_shape(2, 3), TensorOptions().dtype(Dtype::Int32)).unwrap();
        auto original_ptr = destination.as_bytes().data();

        REQUIRE(destination.copy_from(source).is_ok());

        auto dest_data = destination.as_span1d<float>().unwrap();
        for (auto i = 0; i < destination.shape().count(); i++) {
            REQUIRE(dest_data[i] == Catch::Approx(8.0f));
        }
        REQUIRE(original_ptr == destination.as_bytes().data());
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
                make_shape(2, 3),
                TensorOptions().dtype(dtype).device(device)
            );

            REQUIRE(tensor.dtype() == dtype);
            REQUIRE(tensor.device() == device);
        }
    }
}

TEST_CASE("Tensor strides are computed correctly", "[tensor][properties]") {
    SECTION("default row-major strides") {
        auto tensor = Tensor::empty(make_shape(2, 3)).expect("Could not create tensor");

        REQUIRE(tensor.stride().dims() == 2);
        REQUIRE(tensor.stride(0).unwrap() == 3);
        REQUIRE(tensor.stride(1).unwrap() == 1);
    }

    SECTION("custom strides") {
        auto tensor = Tensor::empty(make_shape(2, 3), TensorOptions().stride(make_stride(1, 2)))
                          .expect("Could not create tensor");

        REQUIRE(tensor.stride().dims() == 2);
        REQUIRE(tensor.stride(0).unwrap() == 1);
        REQUIRE(tensor.stride(1).unwrap() == 2);
    }

    SECTION("3D strides") {
        auto tensor = Tensor::empty(make_shape(2, 3, 4)).unwrap();
        REQUIRE(tensor.stride(0).unwrap() == 12);
        REQUIRE(tensor.stride(1).unwrap() == 4);
        REQUIRE(tensor.stride(2).unwrap() == 1);
    }
}

TEST_CASE("Tensor dimensions are tracked correctly", "[tensor][properties]") {
    auto tensor = Tensor::empty(make_shape(2, 3, 4, 5)).expect("Unable to create tensor");

    REQUIRE(tensor.dims() == 4);
    REQUIRE(tensor.shape().dims() == 4);
    REQUIRE(tensor.stride().dims() == 4);
    REQUIRE(tensor.axes().dims() == 4);
}

TEST_CASE("Tensor::empty() detects empty tensors", "[tensor][properties]") {
    SECTION("non-empty tensor") {
        auto tensor = Tensor::empty(make_shape(2, 3)).unwrap();
        REQUIRE_FALSE(tensor.empty());
    }

    SECTION("tensor with zero dimension") {
        auto tensor = Tensor::empty(make_shape(0, 3)).unwrap();
        REQUIRE(tensor.empty());
    }

    SECTION("tensor with multiple zero dimensions") {
        auto tensor = Tensor::empty(make_shape(0, 0)).unwrap();
        REQUIRE(tensor.empty());
    }
}

// ============================================================================
// Tensor Contiguity Tests
// ============================================================================

TEST_CASE("Tensor::is_contiguous detects memory layout", "[tensor][contiguity]") {
    SECTION("default tensors are contiguous") {
        auto tensor = Tensor::empty(make_shape(2, 3)).unwrap();
        REQUIRE(tensor.is_contiguous());
    }

    SECTION("custom strides may not be contiguous") {
        auto tensor =
            Tensor::empty(make_shape(2, 3), TensorOptions().stride(make_stride(1, 2))).unwrap();
        REQUIRE_FALSE(tensor.is_contiguous());
    }
}

TEST_CASE("Tensor::to_contiguous creates contiguous copy", "[tensor][contiguity]") {
    auto tensor =
        Tensor::empty(make_shape(2, 3), TensorOptions().stride(make_stride(1, 2))).unwrap();

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
    auto tensor = Tensor::full(make_shape(3, 3), 1.0).unwrap();
    REQUIRE(tensor.is_contiguous());

    auto contiguous = tensor.to_contiguous().unwrap();
    REQUIRE(contiguous.is_contiguous());

    REQUIRE_THAT(testing::compare_tensors(tensor, contiguous), testing::IsOk());
}

// ============================================================================
// Tensor View Tests
// ============================================================================

TEST_CASE("Tensor::as_view creates view sharing data", "[tensor][view]") {
    auto tensor = Tensor::zeros(make_shape(2, 3)).unwrap();

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
    auto tensor =
        Tensor::full(make_shape(3, 4), 2.5, TensorOptions().stride(make_stride(2, 1))).unwrap();

    auto view = tensor.as_view();
    REQUIRE(view.stride() == tensor.stride());
    REQUIRE_FALSE(view.is_contiguous());
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_CASE("Tensor handles single element", "[tensor][edge-cases]") {
    auto tensor = Tensor::full(make_shape(1), 42.0).unwrap();
    REQUIRE(tensor.size() == 1);
    REQUIRE(tensor.as_span1d<float>().unwrap()[0] == Catch::Approx(42.0f));
}

TEST_CASE("Tensor with large dimensions", "[tensor][edge-cases]") {
    auto tensor = Tensor::empty(make_shape(100, 200)).unwrap();
    REQUIRE(tensor.size() == 20000);
    REQUIRE(tensor.is_contiguous());
}

// ============================================================================
// Tensor Span Access Tests
// ============================================================================

TEST_CASE("Tensor::as_bytes provides byte-level access", "[tensor][span]") {
    auto tensor = Tensor::full(make_shape(2, 3), 42.0f).unwrap();

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
        auto tensor = Tensor::full(make_shape(2, 3), 1.5f).unwrap();
        auto span = tensor.as_span1d<float>().unwrap();

        REQUIRE(span.size() == 6);
        for (size_t i = 0; i < span.size(); i++) {
            REQUIRE(span[i] == Catch::Approx(1.5f));
        }
    }

    SECTION("const version works correctly") {
        auto tensor = Tensor::full(make_shape(4), 2.5f).unwrap();
        const auto& const_tensor = tensor;
        auto span = const_tensor.as_span1d<float>().unwrap();

        REQUIRE(span.size() == 4);
        REQUIRE(span[0] == Catch::Approx(2.5f));
    }

    SECTION("supports int32 dtype") {
        auto tensor = Tensor::full(make_shape(5), 10, TensorOptions().dtype(Dtype::Int32)).unwrap();
        auto span = tensor.as_span1d<int32_t>().unwrap();

        REQUIRE(span.size() == 5);
        for (size_t i = 0; i < span.size(); i++) {
            REQUIRE(span[i] == 10);
        }
    }

    SECTION("supports double dtype") {
        auto tensor =
            Tensor::full(make_shape(3), 3.14, TensorOptions().dtype(Dtype::Float64)).unwrap();
        auto span = tensor.as_span1d<double>().unwrap();

        REQUIRE(span.size() == 3);
        for (size_t i = 0; i < span.size(); i++) {
            REQUIRE(span[i] == Catch::Approx(3.14));
        }
    }

    SECTION("fails with wrong dtype") {
        auto tensor = Tensor::full(make_shape(3), 1.0f).unwrap();
        auto result = tensor.as_span1d<int32_t>();

        REQUIRE(result.is_error());
        REQUIRE(result.err().code() == P10Error::InvalidArgument);
    }

    SECTION("mutable span allows modification") {
        auto tensor = Tensor::zeros(make_shape(4)).unwrap();
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
        auto tensor = Tensor::zeros(make_shape(3, 4)).unwrap();
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
        auto tensor = Tensor::full(make_shape(2, 5), 7.0f).unwrap();
        const auto& const_tensor = tensor;
        auto span = const_tensor.as_span2d<float>().unwrap();

        REQUIRE(span.height() == 2);
        REQUIRE(span.width() == 5);
        REQUIRE(span.row(0)[0] == Catch::Approx(7.0f));
        REQUIRE(span.row(1)[4] == Catch::Approx(7.0f));
    }

    SECTION("supports int64 dtype") {
        auto tensor =
            Tensor::full(make_shape(2, 3), 100, TensorOptions().dtype(Dtype::Int64)).unwrap();
        auto span = tensor.as_span2d<int64_t>().unwrap();

        REQUIRE(span.height() == 2);
        REQUIRE(span.width() == 3);
        REQUIRE(span.row(0)[0] == 100);
    }

    SECTION("fails for non-2D tensor") {
        auto tensor = Tensor::zeros(make_shape(2, 3, 4)).unwrap();
        auto result = tensor.as_span2d<float>();

        REQUIRE(result.is_error());
        REQUIRE(result.err().code() == P10Error::InvalidArgument);
    }

    SECTION("fails with wrong dtype") {
        auto tensor = Tensor::zeros(make_shape(2, 3)).unwrap();
        auto result = tensor.as_span2d<double>();

        REQUIRE(result.is_error());
        REQUIRE(result.err().code() == P10Error::InvalidArgument);
    }
}

TEST_CASE("Tensor::as_span3d converts to 3D span", "[tensor][span]") {
    SECTION("creates 3D span for float tensor") {
        auto tensor = Tensor::zeros(make_shape(2, 3, 4)).unwrap();
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
        auto tensor = Tensor::full(make_shape(2, 2, 3), 5.5f).unwrap();
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
            Tensor::full(make_shape(2, 2, 3), 128, TensorOptions().dtype(Dtype::Uint8)).unwrap();
        auto span = tensor.as_span3d<uint8_t>().unwrap();

        REQUIRE(span.height() == 2);
        REQUIRE(span.width() == 2);
        REQUIRE(span.channels() == 3);
        REQUIRE(span.channel(0, 0)[0] == 128);
    }

    SECTION("row access works correctly") {
        auto tensor = Tensor::zeros(make_shape(3, 4, 2)).unwrap();
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
        auto tensor = Tensor::zeros(make_shape(2, 3)).unwrap();
        auto result = tensor.as_span3d<float>();

        REQUIRE(result.is_error());
        REQUIRE(result.err().code() == P10Error::InvalidArgument);
    }

    SECTION("fails with wrong dtype") {
        auto tensor = Tensor::zeros(make_shape(2, 3, 4)).unwrap();
        auto result = tensor.as_span3d<int32_t>();

        REQUIRE(result.is_error());
        REQUIRE(result.err().code() == P10Error::InvalidArgument);
    }
}

TEST_CASE("Tensor::as_planar_span3d converts to planar 3D span", "[tensor][span]") {
    SECTION("creates planar 3D span for float tensor") {
        auto tensor = Tensor::zeros(make_shape(3, 4, 5)).unwrap();
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
        auto tensor = Tensor::full(make_shape(2, 3, 4), 9.0f).unwrap();
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
            Tensor::full(make_shape(2, 3, 4), 1000, TensorOptions().dtype(Dtype::Uint16)).unwrap();
        auto span = tensor.as_planar_span3d<uint16_t>().unwrap();

        REQUIRE(span.channels() == 2);
        REQUIRE(span.height() == 3);
        REQUIRE(span.width() == 4);
        REQUIRE(span.plane(0).row(0)[0] == 1000);
    }

    SECTION("plane dimensions are correct") {
        auto tensor = Tensor::zeros(make_shape(3, 10, 20)).unwrap();
        auto span = tensor.as_planar_span3d<float>().unwrap();

        for (size_t c = 0; c < 3; c++) {
            auto plane = span.plane(c);
            REQUIRE(plane.height() == 10);
            REQUIRE(plane.width() == 20);
        }
    }

    SECTION("fails for non-3D tensor") {
        auto tensor = Tensor::zeros(make_shape(2, 3, 4, 5)).unwrap();
        auto result = tensor.as_planar_span3d<float>();

        REQUIRE(result.is_error());
        REQUIRE(result.err().code() == P10Error::InvalidArgument);
    }

    SECTION("fails with wrong dtype") {
        auto tensor = Tensor::zeros(make_shape(2, 3, 4)).unwrap();
        auto result = tensor.as_planar_span3d<double>();

        REQUIRE(result.is_error());
        REQUIRE(result.err().code() == P10Error::InvalidArgument);
    }
}

TEST_CASE("Tensor::select_dimension", "[tensor][select_dimension]") {
    SECTION("Select dimension reduces tensor rank") {
        auto tensor = Tensor::full(make_shape(3, 4, 5), 1.0f).unwrap();
        auto selected = tensor.select_dimension(1, 2).unwrap();

        REQUIRE(selected.dims() == 2);
        REQUIRE(selected.shape() == make_shape(3, 5));
        REQUIRE_FALSE(selected.is_contiguous());
    }

    SECTION("Select first dimension") {
        auto tensor = Tensor::full(make_shape(3, 4, 5), 1.0f).unwrap();
        auto selected = tensor.select_dimension(0, 0).unwrap();

        REQUIRE(selected.dims() == 2);
        REQUIRE(selected.shape() == make_shape(4, 5));
        REQUIRE(selected.is_contiguous());
    }

    SECTION("Select last dimension") {
        auto tensor = Tensor::full(make_shape(3, 4, 5), 1.0f).unwrap();
        auto selected = tensor.select_dimension(2, 3).unwrap();

        REQUIRE(selected.dims() == 2);
        REQUIRE(selected.shape() == make_shape(3, 4));
        REQUIRE_FALSE(selected.is_contiguous());
    }

    SECTION("Select dimension verifies data correctness") {
        auto tensor = Tensor::from_range(make_shape(2, 3, 4), Dtype::Float32).unwrap();

        // Tensor data: 0-23 in row-major order
        // Shape: [2, 3, 4], Stride: [12, 4, 1]
        // Layout: [[[0,1,2,3], [4,5,6,7], [8,9,10,11]], [[12,13,14,15], [16,17,18,19], [20,21,22,23]]]

        auto selected = tensor.select_dimension(1, 1).unwrap();
        REQUIRE(selected.dims() == 2);
        REQUIRE(selected.shape() == make_shape(2, 4));

        // Select middle row (index 1) from each batch
        // The view is non-contiguous with stride [12, 1]
        // Starting offset: stride[1] * 1 * 4 bytes = 4 * 4 = 16 bytes (element 4)
        // This gives us a non-contiguous view: first 4 elements starting at offset 4
        // then skip 8 elements (due to stride 12), getting next 4 elements
        auto contiguous = selected.to_contiguous().unwrap();
        auto selected_data = contiguous.as_span1d<float>().unwrap();
        REQUIRE(selected_data[0] == Catch::Approx(4.0f));
        REQUIRE(selected_data[1] == Catch::Approx(5.0f));
        REQUIRE(selected_data[2] == Catch::Approx(6.0f));
        REQUIRE(selected_data[3] == Catch::Approx(7.0f));
        REQUIRE(selected_data[4] == Catch::Approx(16.0f));
        REQUIRE(selected_data[5] == Catch::Approx(17.0f));
        REQUIRE(selected_data[6] == Catch::Approx(18.0f));
        REQUIRE(selected_data[7] == Catch::Approx(19.0f));
    }

    SECTION("Select dimension on 2D tensor produces 1D") {
        auto tensor = Tensor::from_range(make_shape(4, 5), Dtype::Int32).unwrap();
        auto selected = tensor.select_dimension(0, 2).unwrap();

        REQUIRE(selected.dims() == 1);
        REQUIRE(selected.shape() == make_shape(5));

        // Should get row 2: elements 10-14
        auto selected_data = selected.as_span1d<int32_t>().unwrap();
        for (int i = 0; i < 5; i++) {
            REQUIRE(selected_data[i] == 10 + i);
        }
    }

    SECTION("Select dimension shares data with original") {
        auto tensor = Tensor::zeros(make_shape(3, 4)).unwrap();
        auto selected = tensor.select_dimension(0, 1).unwrap();

        // Modify selected view
        selected.as_span1d<float>().unwrap()[0] = 42.0f;

        // Original tensor should reflect the change
        auto tensor_data = tensor.as_span1d<float>().unwrap();
        REQUIRE(tensor_data[4] == Catch::Approx(42.0f));
    }

    SECTION("Select boundary indices") {
        auto tensor = Tensor::full(make_shape(5, 3, 2), 7.0f).unwrap();

        // First index (index 0) - contiguous
        auto first = tensor.select_dimension(0, 0).unwrap();
        REQUIRE(first.shape() == make_shape(3, 2));
        REQUIRE(first.is_contiguous());

        // Last index (index 4) - contiguous
        auto last = tensor.select_dimension(0, 4).unwrap();
        REQUIRE(last.shape() == make_shape(3, 2));
        REQUIRE(last.is_contiguous());
    }

    SECTION("Select dimension with different dtypes") {
        SECTION("int64") {
            auto tensor =
                Tensor::full(make_shape(3, 4), 100, TensorOptions().dtype(Dtype::Int64)).unwrap();
            auto selected = tensor.select_dimension(1, 2).unwrap();
            REQUIRE(selected.dtype() == Dtype::Int64);
            REQUIRE(selected.shape() == make_shape(3));
        }

        SECTION("uint8") {
            auto tensor =
                Tensor::full(make_shape(2, 5, 3), 255, TensorOptions().dtype(Dtype::Uint8))
                    .unwrap();
            auto selected = tensor.select_dimension(2, 1).unwrap();
            REQUIRE(selected.dtype() == Dtype::Uint8);
            REQUIRE(selected.shape() == make_shape(2, 5));
        }
    }

    SECTION("Multiple consecutive selections") {
        auto tensor = Tensor::from_range(make_shape(4, 3, 2), Dtype::Float32).unwrap();
        // Tensor shape: [4, 3, 2], Stride: [6, 2, 1]
        // Layout: 0,1, 2,3, 4,5, | 6,7, 8,9, 10,11, | 12,13, 14,15, 16,17, | 18,19, 20,21, 22,23

        // First selection: [4, 3, 2] -> [3, 2] at index 1
        // Offset: 6 * 1 * 4 = 24 bytes (element 6)
        // New shape: [3, 2], New stride: [2, 1]
        auto selected1 = tensor.select_dimension(0, 1).unwrap();
        REQUIRE(selected1.shape() == make_shape(3, 2));

        // Second selection: [3, 2] -> [2] at index 2
        // Offset from selected1 start: 2 * 2 * 4 = 16 bytes (4 elements from selected1 start)
        // Absolute position: element 6 + 4 = element 10
        auto selected2 = selected1.select_dimension(0, 2).unwrap();
        REQUIRE(selected2.shape() == make_shape(2));

        // Should get elements from tensor[1, 2, :] which is elements 10 and 11
        auto data = selected2.as_span1d<float>().unwrap();
        REQUIRE(data[0] == Catch::Approx(10.0f));
        REQUIRE(data[1] == Catch::Approx(11.0f));
    }

    SECTION("Error: invalid dimension index") {
        auto tensor = Tensor::full(make_shape(3, 4, 5), 1.0f).unwrap();

        auto result1 = tensor.select_dimension(3, 0);
        REQUIRE(result1.is_error());
        REQUIRE(result1.err().code() == P10Error::InvalidArgument);

        auto result2 = tensor.select_dimension(-1, 0);
        REQUIRE(result2.is_error());
        REQUIRE(result2.err().code() == P10Error::InvalidArgument);
    }

    SECTION("Error: index out of bounds") {
        auto tensor = Tensor::full(make_shape(3, 4, 5), 1.0f).unwrap();

        // Index too large
        auto result1 = tensor.select_dimension(1, 4);
        REQUIRE(result1.is_error());
        REQUIRE(result1.err().code() == P10Error::InvalidArgument);

        auto result2 = tensor.select_dimension(2, 5);
        REQUIRE(result2.is_error());
        REQUIRE(result2.err().code() == P10Error::InvalidArgument);

        // Negative index
        auto result3 = tensor.select_dimension(0, -1);
        REQUIRE(result3.is_error());
        REQUIRE(result3.err().code() == P10Error::InvalidArgument);
    }

    SECTION("Stride preservation") {
        auto tensor = Tensor::full(make_shape(4, 5, 6), 1.0f).unwrap();

        // Original strides: [30, 6, 1]
        REQUIRE(tensor.stride(0).unwrap() == 30);
        REQUIRE(tensor.stride(1).unwrap() == 6);
        REQUIRE(tensor.stride(2).unwrap() == 1);

        auto selected = tensor.select_dimension(1, 2).unwrap();
        // After selecting dim 1, strides should be [30, 1]
        REQUIRE(selected.stride(0).unwrap() == 30);
        REQUIRE(selected.stride(1).unwrap() == 1);
    }
}

TEST_CASE("Tensor::reshape", "[tensor][reshape]") {
    SECTION("Valid shapes") {
        auto tensor = Tensor::from_range(make_shape(2, 3, 4), Dtype::Float32).unwrap();
        size_t tensor_size = tensor.size();
        // Reshape to (4, 3, 2)
        REQUIRE_THAT(tensor.reshape(make_shape(4, 3, 2)), testing::IsOk());
        REQUIRE(tensor.shape() == make_shape(4, 3, 2));
        REQUIRE(tensor.size() == tensor_size);
        // Reshape to (6, 4)
        REQUIRE_THAT(tensor.reshape(make_shape(6, 4)), testing::IsOk());
        REQUIRE(tensor.shape() == make_shape(6, 4));
        REQUIRE(tensor.size() == tensor_size);
        // Reshape to (24,)
        REQUIRE_THAT(tensor.reshape(make_shape(24)), testing::IsOk());
        REQUIRE(tensor.shape() == make_shape(24));
        REQUIRE(tensor.size() == tensor_size);
    }

    SECTION("Valid shapes non-contiguous tensor") {
        auto tensor =
            Tensor::full(make_shape(2, 3), 1.0f, TensorOptions().stride(make_stride(1, 2)))
                .unwrap();
        size_t tensor_size = tensor.size();
        // Reshape to (3, 2)
        REQUIRE_THAT(tensor.reshape(make_shape(3, 2)), testing::IsOk());
        REQUIRE(tensor.shape() == make_shape(3, 2));
        REQUIRE(tensor.size() == tensor_size);
    }

    SECTION("Invalid shapes") {
        auto tensor = Tensor::from_range(make_shape(2, 3, 4), Dtype::Float32).unwrap();

        // Total elements mismatch
        REQUIRE(tensor.reshape(make_shape(5, 5)) == P10Error::InvalidArgument);
        REQUIRE(tensor.reshape(make_shape(10)) == P10Error::InvalidArgument);
    }

    SECTION("Invalid shapes on empty tensor") {
        auto tensor = Tensor::empty(make_shape(0, 3)).unwrap();
        size_t tensor_size = tensor.size();

        // Reshape to (0, 1)
        REQUIRE_THAT(tensor.reshape(make_shape(0, 1)), testing::IsOk());
        REQUIRE(tensor.shape() == make_shape(0, 1));
        REQUIRE(tensor.size() == tensor_size);
        // Invalid reshape
        REQUIRE_THAT(tensor.reshape(make_shape(1, 1)), testing::IsError(P10Error::InvalidArgument));
    }

    SECTION("Reshape shares data with original tensor") {
        auto tensor = Tensor::from_range(make_shape(2, 3), Dtype::Float32).unwrap();
        REQUIRE_THAT(tensor.reshape(make_shape(3, 2)), testing::IsOk());

        // Modify reshaped tensor
        tensor.as_span1d<float>().unwrap()[0] = 99.0f;

        // Original tensor should reflect the change
        auto original_data = tensor.as_span1d<float>().unwrap();
        REQUIRE(original_data[0] == Catch::Approx(99.0f));
    }

    SECTION("Invalid shape non-contiguous tensor") {
        auto tensor =
            Tensor::full(make_shape(2, 3), 1.0f, TensorOptions().stride(make_stride(1, 2)))
                .unwrap();

        // Attempt to reshape to incompatible shape
        REQUIRE_THAT(tensor.reshape(make_shape(4, 2)), testing::IsError(P10Error::InvalidArgument));
    }
}

TEST_CASE("Tensor::transpose", "[tensor][transpose]") {
    SECTION("2D tensor transpose") {
        auto type = GENERATE(Dtype::Float32, Dtype::Int64, Dtype::Uint8);
        DYNAMIC_SECTION("Testing transpose with type " << to_string(type)) {
            auto tensor = Tensor::from_range(make_shape(2, 3), type).unwrap();
            Tensor transposed;
            REQUIRE(tensor.transpose(transposed).is_ok());

            REQUIRE(transposed.shape() == make_shape(3, 2));
            REQUIRE(transposed.stride(0).unwrap() == 2);
            REQUIRE(transposed.stride(1).unwrap() == 1);

            REQUIRE(tensor.stride(0).unwrap() == 3);
            REQUIRE(tensor.stride(1).unwrap() == 1);

            // Verify data correctness
            tensor.visit([&](const auto& type) {
                using T = typename std::decay_t<decltype(type)>::element_type;
                auto original_data = tensor.as_span1d<T>().unwrap();
                auto transposed_data = transposed.as_span1d<T>().unwrap();
                for (size_t i = 0; i < 2; i++) {
                    for (size_t j = 0; j < 3; j++) {
                        REQUIRE(transposed_data[j * 2 + i] == original_data[i * 3 + j]);
                    }
                }
            });
        }
    }

    SECTION("Invalid cases") {
        SECTION("Non 2D Tensor") {
            auto tensor = Tensor::from_range(make_shape(2, 3, 4), Dtype::Float32).unwrap();
            Tensor transposed;
            auto result = tensor.transpose(transposed);
            REQUIRE_THAT(result, testing::IsError(P10Error::InvalidArgument));
            REQUIRE(transposed.empty());
        }

        SECTION("Non contiguous Tensor") {
            auto tensor =
                Tensor::full(make_shape(2, 3), 1.0f, TensorOptions().stride(make_stride(1, 2)))
                    .unwrap();
            Tensor transposed;
            auto result = tensor.transpose(transposed);
            REQUIRE_THAT(result, testing::IsError(P10Error::NotImplemented));
            REQUIRE(transposed.empty());
        }

        SECTION("Invalid device") {
            auto tensor = Tensor::from_data(
                nullptr,
                make_shape(2, 3),
                TensorOptions().dtype(Dtype::Float32).device(Device::Cuda)
            );

            Tensor transposed;
            auto result = tensor.transpose(transposed);
            REQUIRE_THAT(result, testing::IsError(P10Error::NotImplemented));
            REQUIRE(transposed.empty());
        }
    }
}
}  // namespace p10
