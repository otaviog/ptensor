---
name: lang-binder
description "Bind the C-API to other languages"
---

# Make and maintain bindings for other languages

Use this skill when the user to create or update the bindings for one language. The binding architecture uses only the C-API and show follow the C-API maintainer.

## Directories

The C library is under `src/c`.
The bindings are in `bindings`

For Python binding:

* It's here: `bindings/python`
* Use uv
* `uv sync` should also build the library too

For typescript bindings:

* It's here: `bindings/typescript`
* Uses bun as default runtime
* Binds to capi
* Use vitest as testing framework
* Future we may add support to NodeJS and Webassembly
* Dont import .js, leave without extension.
