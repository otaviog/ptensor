# target_link_whole_archive(<target> <lib>)
#
# Link a static archive into <target> with whole-archive semantics so every
# symbol in <lib> is exported from the resulting shared library, not just
# those referenced by <target>'s own translation units.
#
# Internally calls target_link_libraries(<target> PUBLIC <lib>) for dep
# tracking and header propagation, then appends the platform-specific
# force-load flag to pull in the full archive.
function(target_link_whole_archive target lib)
    # PRIVATE: symbols are embedded via whole-archive; consumers link the
    # shared lib directly and don't need to know about the sub-archives.
    target_link_libraries(${target} PRIVATE ${lib})
    # Forward include dirs from the bundled lib to consumers of the shared lib.
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:$<TARGET_PROPERTY:${lib},INTERFACE_INCLUDE_DIRECTORIES>>
    )
    if(MSVC)
        # cl.exe or clang-cl: linker accepts /WHOLEARCHIVE directly.
        target_link_options(${target} PRIVATE "SHELL:/WHOLEARCHIVE:$<TARGET_FILE:${lib}>")
    elseif(WIN32)
        # Plain Clang on Windows uses lld-link; pass /WHOLEARCHIVE via -Wl, so
        # clang forwards it as a single argument to lld-link.
        target_link_options(${target} PRIVATE
            "SHELL:-Wl,/WHOLEARCHIVE:$<TARGET_FILE:${lib}>")
    elseif(APPLE)
        target_link_options(${target} PRIVATE "-Wl,-force_load,$<TARGET_FILE:${lib}>")
    else()
        target_link_options(${target} PRIVATE
            "-Wl,--whole-archive" "$<TARGET_FILE:${lib}>" "-Wl,--no-whole-archive")
    endif()
endfunction()
