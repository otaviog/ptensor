#pragma once

#include <string_view>

#ifdef __cpp_exceptions
    #include <stdexcept>
#else
    #include <cstdlib>
    #include <iostream>
#endif

namespace p10::detail {
[[noreturn]] inline void panic(const std::string_view& error_message) {
#ifdef __cpp_exceptions
    throw std::runtime_error(error_message.data());
#else
    std::cerr << error_message << std::endl;
    std::abort();
#endif
}
}  // namespace p10::detail