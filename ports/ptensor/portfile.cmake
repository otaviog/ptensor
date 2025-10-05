vcpkg_from_git(
    OUT_SOURCE_PATH SOURCE_PATH
    URL "https://github.com/otaviog/ptensor.git"
    REF "f7d44b0f3add485b92884e5800f9326d799fd358"
    SHA512 f46ffcd9a3beab1b394ca6847d7be43d2d5b90a66047bb859c36bb0146cf958e
)

set(FEATURE_io OFF)
if("io" IN_LIST FEATURES)
    set(FEATURE_io ON)
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DBUILD_TESTS=OFF
        -DWITH_IO=${FEATURE_io}
)

vcpkg_cmake_build()

# Install the library
vcpkg_cmake_install()

# Fix the CMake targets export
vcpkg_cmake_config_fixup(
    PACKAGE_NAME ptensor
    CONFIG_PATH lib/cmake/ptensor
)

# Copy license
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")

# Remove debug includes and other cleanup
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

# Verify that the library was built correctly
file(GLOB PTENSOR_LIBS "${CURRENT_PACKAGES_DIR}/lib/*ptensor*")
if(NOT PTENSOR_LIBS)
    message(FATAL_ERROR "ptensor library not found")
endif()