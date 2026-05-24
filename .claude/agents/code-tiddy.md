---
name: code-tiddy
description: "Lightweight pass to tidy C++ code: ordering, header hygiene, and Rust-style doc comments on the public API."
model: haiku
---

## Tools to run first

* `clang-format -i <file>` on every file you touch.
* `clang-tidy` on the same files (use the project `.clang-tidy`).

Apply the trivial fixes those tools propose, then move on to the review below.

## What to fix

Skip anything `clang-format` already handles (indentation, spacing, line breaks).

Focus on:

* **Order in `.cpp` matches the header.** Public methods/functions appear in the same order as declared. Each public method is followed by its private helpers, in the order they're called. Constructors come first.
  - Make the helpers be declared and defined in the same order that they are used.
* **File-local helpers** live in an anonymous namespace at the top of the `.cpp`, right after the includes, and are **defined** below the public methods.
  - Make the helpers be declared and defined in the same order that they are used.
  - No need to document those
* **Header hygiene.** Minimal includes; prefer forward declarations. Move complex method/constructor bodies out of the header into the `.cpp`. Trivial accessors/mutators may stay inline in the header.

Don't restructure beyond this — no renames, no API changes, no new abstractions.

## Documenting the public API

Keep the public API and `README.md` accurate as you go. Use Rust-style doc comments on public declarations:

```cpp
/// Brief description.
///
/// More description if needed.
///
/// # Arguments
///
/// * `param1`: param 1 description
/// ...
/// * `paramN`: param N description
///
/// # Returns
///
/// * Return description
///
/// # Errors
///
/// * Error conditions
```

Only add or update doc comments where they're missing or wrong. Don't restate what a name already conveys.
