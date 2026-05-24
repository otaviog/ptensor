---
name: to_string
description: "Create good to_string methods in C++"
---

# When

Use this skill when creating a `std::string to_string(const Foo& foo)` function in C++.

# How

- Use `std::format`.
- Free function in the `p10` (or relevant) namespace — not a member function.
- Definition goes in the `.cpp` if one exists; otherwise inline in the header.
- For nested types, recursively call their `to_string`.
- Skip private/internal cache fields that don't contribute to the type's identity.

# Format

Except for `Tensor` (which has its own format), use a JSON-compatible-ish layout:

```
{ fieldX: value1, fieldY: value2, complexField: { fieldA: value3 }, listField: [a, b, c] }
```

Conventions:

- **Keys**: unquoted identifiers, `snake_case`.
- **Strings**: double-quoted (`"foo"`).
- **Numbers / bools**: bare (`42`, `1.5`, `true`).
- **Lists / vectors / spans**: `[item1, item2, item3]`, comma + space between items.
- **Nested structs**: `{ ... }` via their own `to_string`.
- **`std::optional`**: value via inner `to_string` when set, literal `null` when empty.
- **Enums**: the enumerator name as a bare identifier (e.g. `Float32`, not `"Float32"` or `3`).

The goal is "shareable as JSON-ish text" — not strict JSON. Don't add escaping logic beyond what the embedded values need.

# Example

Header:

```cpp
// foo.hpp
namespace p10 {
struct Foo {
    int count;
    std::string name;
    std::optional<float> threshold;
};

std::string to_string(const Foo& foo);
}
```

Implementation:

```cpp
// foo.cpp
#include <format>

std::string p10::to_string(const Foo& foo) {
    return std::format(
        R"({{ count: {}, name: "{}", threshold: {} }})",
        foo.count,
        foo.name,
        foo.threshold ? std::format("{}", *foo.threshold) : "null"
    );
}
```

# Verify

- Add a Catch2 test under the module's `tests/` directory.
- Assert the exact string for one representative value, and exercise at least one edge case (empty list, `nullopt`, nested type).

```cpp
TEST_CASE("Foo::to_string", "[foo]") {
    Foo foo{.count = 2, .name = "bar", .threshold = 0.5f};
    REQUIRE(to_string(foo) == R"({ count: 2, name: "bar", threshold: 0.5 })");
}
```
