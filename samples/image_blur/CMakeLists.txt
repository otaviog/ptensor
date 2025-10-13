cmake_minimum_required(VERSION 3.31)
project(sample-blur)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ptensor REQUIRED)
add_library(sample_blur SHARED blur_lib.cpp)
target_link_libraries(sample_blur p10::ptensor p10::io p10::op)
