add_executable(p10_wasm module.cpp)
target_link_libraries(p10_wasm ptensor)

set_target_properties(p10_wasm PROPERTIES
    LINK_FLAGS "-lembind -s MODULARIZE=1 -s EXPORT_NAME='createP10' -s ALLOW_MEMORY_GROWTH=1 -s WASM=1 -s EXPORTED_FUNCTIONS=['_malloc','_free'] -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap','getValue','setValue','HEAP8','HEAP16','HEAP32','HEAPU8','HEAPU16','HEAPU32','HEAPF32','HEAPF64'] -s EXPORT_ES6=1"
    OUTPUT_NAME "p10"
    RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_SOURCE_DIR}/wasm/build
    SUFFIX ".js"
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/p10.js
    ${CMAKE_CURRENT_BINARY_DIR}/p10.wasm
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
