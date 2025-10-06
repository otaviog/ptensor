function(ptensor_target_options target folder)
    target_compile_options(${target} PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
    )

    get_target_property(target_type ${target} TYPE)
    if(NOT WIN32 AND target_type STREQUAL "STATIC_LIBRARY")
        target_compile_options(${target} PRIVATE -fPIC)
    endif()

    set_target_properties(${target} PROPERTIES
        FOLDER ${folder})
endfunction()
