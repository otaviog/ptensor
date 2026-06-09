---
name: tensor-handling
description: "How to read, write, and allocate data with p10::Tensor: output params, visit vs match, spans vs accessors."
---

# Working with p10::Tensor

Use this skill when writing or reviewing code that reads from or writes to
`p10::Tensor` (most code under `src/op/`, `src/core/`, `src/infer/`).

`Tensor` is a typed, multi-dimensional buffer. Its element type is a runtime
`Dtype`, so element access goes through one of the type-dispatch helpers below
(`visit` / `match`) rather than a compile-time `T`.

## Output parameters: take `Tensor&`, fill it with `create`

Operations do not return a `Tensor`. They take the output by reference and
fill it. The caller owns the storage; the op shapes it.

```c++
p10::P10Error add(const p10::Tensor& a, const p10::Tensor& b, p10::Tensor& out) {
    P10_RETURN_IF_ERROR(out.create(make_shape(rows, cols), a.dtype()));
    // ... fill `out` ...
    return p10::P10Error::Ok;
}
```

* `create(shape, options)` reuses the existing allocation when the byte size
  already fits, so reusing an `out` tensor across calls avoids reallocation.
* The second argument is a `TensorOptions`; a bare `Dtype` converts to it
  implicitly (`a.dtype()` above, or `TensorOptions(Dtype::Float64)`).
* `make_shape(a, b, ...)` (fixed-arg overloads) returns a `Shape` directly. The
  `make_shape(std::span)` overload returns a `P10Result<Shape>` — unwrap or
  propagate its error.
* Wrap fallible calls in `P10_RETURN_IF_ERROR` (returns the `P10Error`) or
  `P10_RETURN_ERR_IF_ERROR` (returns `Err(...)` from a `P10Result` function).

A Tensor is tipically return when the function is creating a view from it.

## Choosing how to access elements

Decide along two axes: do you need a **flat byte span** or a **typed accessor**,
and is the data **contiguous** or possibly **strided**?

### `tensor.visit` — flat span over contiguous data

`visit` hands the lambda a `std::span<scalar_t>` over the whole buffer, already
typed. It **asserts the tensor is contiguous**. Use it for elementwise work or
when you compute multi-dimensional indices yourself.

```c++
tensor.visit([&](auto span) {
    using scalar_t = typename decltype(span)::value_type;
    for (auto& v : span) { /* ... */ }
});
```

The lambda is instantiated for every dtype, so keep the body generic. To force a
common accumulator/result type, cast inside (e.g. `static_cast<double>(v)`) and
write to a separately-allocated `double` output — see `op::mean` in
`src/op/statistics.cpp`.

### `dtype().match` — pick the concrete type, then use a typed accessor

`match` gives you a type tag instead of a span, so you can call the typed
accessor that fits the rank you need. This is the path for multi-dimensional or
strided access.

```c++
return input.dtype().match([&](auto type_tag) -> p10::P10Error {
    using scalar_t = typename decltype(type_tag)::type;

    auto in = input.as_accessor3d<scalar_t>();   // P10Result
    if (in.is_error()) { return in.error(); }
    // ... use in.unwrap() ...
    return p10::P10Error::Ok;
});
```

Every `match` branch must return the same type, so annotate the lambda's return
type (`-> P10Error`). There is also a two-lambda overload
`match(int_matcher, float_matcher)` when integer and floating-point cases need
different code.

### Spans (contiguous) vs accessors (strided)

* **Spans** — `as_span1d`, `as_span2d`, `as_span3d`, `as_span4d`. Require the
  tensor to be **contiguous**; `as_span2d`/`3d`/`4d` also require the matching
  rank. Fastest, simplest indexing.
* **Accessors** — `as_accessor1d`, `as_accessor2d`, `as_accessor3d` (no 4D).
  Honor `stride`, so they work on **non-contiguous** views.

All of these return a `P10Result<...>` and need a concrete `scalar_t`, so reach
them via `match` (or pass an explicit type when you already know it).

A common, valid shortcut: if you only intend to support contiguous input,
check `is_contiguous()` (or `to_contiguous()`) up front and use spans, returning
an error otherwise.

## Shape and stride

* `tensor.shape().as_span()` / `tensor.stride().as_span()` give
  `std::span<const int64_t>` for index math when the rank is known at runtime.
* `tensor.dims()`, `tensor.size()` (element count), `tensor.shape(axis)` for
  individual values.

## Before writing a new op, check `p10::op`

Reuse what exists in the `p10::op` namespace (`src/op/`) — resize, crop, blur,
fft, elementwise, statistics, stack, etc. — rather than re-deriving access
patterns.
