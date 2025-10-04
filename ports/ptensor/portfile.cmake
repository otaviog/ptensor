vcpkg_from_git(
    OUT_SOURCE_PATH SOURCE_PATH
    URL "https://github.com/otaviog/ptensor.git"
    REF "${VERSION}"
    SHA512 0  # This will be updated when you use the port
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DBUILD_TESTS=OFF
        -DWITH_IO=ON
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