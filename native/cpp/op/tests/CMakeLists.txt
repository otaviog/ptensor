add_executable(test_op
    test_elemwise.cpp
    test_fft.cpp
    test_resize.cpp
    test_blur.cpp
)

target_link_libraries(test_op PRIVATE
    ptensor_op
    ptensor_testing
    Catch2::Catch2WithMain
)

target_compile_options(test_op PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

set_target_properties(test_op PROPERTIES
    FOLDER Tests
)

catch_discover_tests(test_op)
