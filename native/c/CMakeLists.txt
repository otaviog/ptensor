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
    ${_INCLUDE_DIR}/tensor_wrapper.hpp
    ${_INCLUDE_DIR}/ptensor_dtype.h
    ${_INCLUDE_DIR}/ptensor_device.h
    ${_INCLUDE_DIR}/ptensor_int_types.h)

target_sources(ptensor_capi
    PRIVATE
    src/ptensor_error.cpp
    src/ptensor_tensor.cpp
    src/dtype_wrapper.hpp
    src/update_error_state.hpp
    src/ptensor_dtype.cpp
    )
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
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${_INCLUDE_DIR}
)
target_link_libraries(ptensor_capi ptensor)
set_target_properties(ptensor_capi PROPERTIES
        FOLDER "Core")

install(TARGETS ptensor_capi
    EXPORT ptensorTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    FILE_SET public_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
