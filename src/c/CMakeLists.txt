set(_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/ptensor)

add_library(ptensor_capi SHARED)
target_sources(ptensor_capi
    PUBLIC
    FILE_SET public_headers TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
    FILES
    ${_INCLUDE_DIR}/config.h
    ${_INCLUDE_DIR}/ptensor_error.h
    ${_INCLUDE_DIR}/ptensor_tensor.h
    ${_INCLUDE_DIR}/ptensor_dtype.h
    ${_INCLUDE_DIR}/ptensor_device.h
    ${_INCLUDE_DIR}/ptensor_int_types.h)

target_sources(ptensor_capi
    PRIVATE
    ptensor_error.cpp
    ptensor_tensor.cpp
    dtype_wrapper.hpp
    update_error_state.hpp
    ptensor_dtype.cpp
    )

if (WITH_INFER)
    target_sources(ptensor_capi
        PUBLIC
        FILE_SET public_headers TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
        FILES
        ${_INCLUDE_DIR}/ptensor_infer.h)
    target_sources(ptensor_capi
        PRIVATE
        infer_wrapper.hpp
        ptensor_infer.cpp)
    target_link_libraries(ptensor_capi PRIVATE ptensor_infer)
    target_compile_definitions(ptensor_capi PRIVATE PTENSOR_WITH_INFER)
endif()

if (WITH_MEDIA)
    target_sources(ptensor_capi
        PUBLIC
        FILE_SET public_headers TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
        FILES
        ${_INCLUDE_DIR}/ptensor_media.h)
    target_sources(ptensor_capi
        PRIVATE
        media_wrapper.hpp
        ptensor_media.cpp)
    target_link_libraries(ptensor_capi PRIVATE ptensor_media)
    target_compile_definitions(ptensor_capi PRIVATE PTENSOR_WITH_MEDIA)
endif()

target_compile_definitions(
    ptensor_capi
    PRIVATE
    PTENSOR_BUILD_SHARED
)
target_include_directories(ptensor_capi
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE
    ${_INCLUDE_DIR}/capi
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${_INCLUDE_DIR}
)
target_link_libraries(ptensor_capi PUBLIC $<BUILD_INTERFACE:ptensor>)

# FFmpeg's NASM assembly isn't compiled with -fPIC. A version script hides FFmpeg's
# internal symbols (making them LOCAL), which lets the linker resolve PC32 relocations
# within the same shared object boundary.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(ptensor_capi PRIVATE
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/ptensor_capi.version"
    )
endif()

if(APPLE)
    # capi links the core archive and the optional module archives, which each
    # re-propagate core and the webp/sharpyuv deps, so they appear twice on the
    # link line. The duplicates are harmless (linker ignores them); silence the
    # warning, matching ptensor_shared.
    target_link_options(ptensor_capi PRIVATE -Wl,-no_warn_duplicate_libraries)
endif()

if(APPLE)
    # onnxruntime is installed next to libptensor_capi.dylib, so @loader_path
    # resolves it at runtime. Build with the install rpath directly so the install
    # step does not rewrite it (the rewrite is not idempotent on incremental
    # installs and errors out when the dylib is already up-to-date).
    set_target_properties(ptensor_capi PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "@loader_path")
endif()

# Install the target
install(TARGETS ptensor_capi
    EXPORT ptensorTargets
    LIBRARY DESTINATION ${PTENSOR_INSTALL_LIB_DIR}
    RUNTIME DESTINATION ${PTENSOR_INSTALL_BIN_DIR}
    FILE_SET public_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
