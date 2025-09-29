add_subdirectory(core)

set(WITH_IO OFF CACHE BOOL "Build IO library")

if (BUILD_TESTS)
  set(WITH_IO TRUE)
endif()

if (WITH_IO)
  #find_package(ZLIB REQUIRED)
  #add_subdirectory(lib/cnpy)
  #set(StbImage_DIR lib/stb)
  # find_package(StbImage REQUIRED)
endif (WITH_IO)
