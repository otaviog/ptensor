cmake_minimum_required(VERSION 3.15)

# Tensor-core symbols to export to JavaScript.
set(PTENSOR_WASM_EXPORTED_FUNCS
    "_malloc"
    "_free"
    "_p10_get_last_error_message"
    "_p10_from_data"
    "_p10_from_data_strided"
    "_p10_destroy"
    "_p10_get_size"
    "_p10_get_size_bytes"
    "_p10_get_dtype"
    "_p10_get_shape"
    "_p10_get_stride"
    "_p10_get_ndim"
    "_p10_get_data"
    "_p10_is_empty"
    "_p10_dtype_to_string"
    "_p10_dtype_size_bytes"
)
string(JOIN "," _EXPORTED_FUNCS_CSV ${PTENSOR_WASM_EXPORTED_FUNCS})

set(PTENSOR_WASM_EXPORTED_RUNTIME_METHODS
    "stackAlloc"
    "stackSave"
    "stackRestore"
    "getValue"
    "setValue"
    "UTF8ToString"
    "HEAPU8"
)
string(JOIN "," _EXPORTED_RUNTIME_CSV ${PTENSOR_WASM_EXPORTED_RUNTIME_METHODS})

# ptensor_capi is built as STATIC when EMSCRIPTEN is set (see src/c/CMakeLists.txt).
add_executable(ptensor_wasm
    ${CMAKE_SOURCE_DIR}/src/c/src/ptensor_tensor.cpp
    ${CMAKE_SOURCE_DIR}/src/c/src/ptensor_error.cpp
    ${CMAKE_SOURCE_DIR}/src/c/src/ptensor_dtype.cpp
)

target_include_directories(ptensor_wasm
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src/c/include
    ${CMAKE_SOURCE_DIR}/src/c/src
)

target_link_libraries(ptensor_wasm PRIVATE ptensor)

target_link_options(ptensor_wasm PRIVATE
    "SHELL:-s WASM=1"
    "SHELL:-s ALLOW_MEMORY_GROWTH=1"
    "SHELL:-s MODULARIZE=1"
    "SHELL:-s EXPORT_NAME=PTensorModule"
    "SHELL:-s EXPORTED_FUNCTIONS=[${_EXPORTED_FUNCS_CSV}]"
    "SHELL:-s EXPORTED_RUNTIME_METHODS=[${_EXPORTED_RUNTIME_CSV}]"
)

set_target_properties(ptensor_wasm PROPERTIES
    OUTPUT_NAME "ptensor"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/js"
)
