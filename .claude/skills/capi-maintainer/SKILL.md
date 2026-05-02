---
name: capi-maintainer
description "Maintain the C api as it should be: clean, small, and concise so we expose what the user needs to run with other apis
---

# Maintain the C API.

Use this skill when the user asks to add a new binding to the C library from the C++ one, when the user whats to improve the current C library, or fix some bug.

## Goals

**The whole ideia of PTensor is to NOT offer tensor manipulation.** It just a library that offer data handling so Numpy/Pytorch/Tensorflow due the actual operation. 
We just get the data, and tells or manipulate its type, shape, and stride. We have some more complex changes, but they are for the C++ only part.

## Directory

The C library is under `src/c`.

## Style

* Use opaque pointers
* Use Rust style naming
* Avoid using external headers

Names are like:

`p10_<module>_<function-name>`, except for the core library which are `p10_<function-name>`.

