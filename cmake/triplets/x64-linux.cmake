set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_CMAKE_SYSTEM_NAME Linux)

# Required for linking static libraries into shared libraries (ptensor_media)
set(VCPKG_CXX_FLAGS "-fPIC")
set(VCPKG_C_FLAGS "-fPIC")
