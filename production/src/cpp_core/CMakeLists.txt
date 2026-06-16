cmake_minimum_required(VERSION 3.14)
project(cpp_features)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11
find_package(pybind11 REQUIRED)

# Add our pybind11 extension module
pybind11_add_module(cpp_features 
    cpp_feature_pipeline.cpp
    cpp_dsa.cpp
)

# Set compile flags for performance
if(MSVC)
    target_compile_options(cpp_features PRIVATE /O2 /fp:fast)
else()
    target_compile_options(cpp_features PRIVATE -O3 -ffast-math -march=native)
endif()
