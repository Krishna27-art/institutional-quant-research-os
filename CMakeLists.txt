cmake_minimum_required(VERSION 3.15)
project(NiftyQuant VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags for production execution engine
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -flto")
endif()

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    # Fetch pybind11 if not installed
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Find spdlog for logging
find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.12.0
    )
    FetchContent_MakeAvailable(spdlog)
endif()

# Execution Engine Library (static)
add_library(execution_engine STATIC
    engine/order_book.cpp
    engine/execution_engine.cpp
)
target_include_directories(execution_engine PUBLIC engine/)
target_link_libraries(execution_engine PUBLIC Python3::Python spdlog::spdlog)

# pybind11 Module
pybind11_add_module(niftyquant_cpp engine/bindings.cpp)
target_link_libraries(niftyquant_cpp PRIVATE execution_engine)

# Install
install(TARGETS niftyquant_cpp
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}
)
