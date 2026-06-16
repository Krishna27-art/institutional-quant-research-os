cmake_minimum_required(VERSION 3.14)
project(cpp_zmq_consumer)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11
find_package(pybind11 REQUIRED)

# In a real environment, we would also find ZeroMQ
# find_package(ZeroMQ REQUIRED)

# Add our pybind11 extension module
pybind11_add_module(cpp_zmq_consumer 
    zmq_consumer.cpp
)

pybind11_add_module(cpp_timeseries_db
    cpp_timeseries_db.cpp
)

# target_link_libraries(cpp_zmq_consumer PRIVATE ZeroMQ)

# Set compile flags for performance
if(MSVC)
    target_compile_options(cpp_zmq_consumer PRIVATE /O2 /fp:fast)
    target_compile_options(cpp_timeseries_db PRIVATE /O2 /fp:fast)
else()
    target_compile_options(cpp_zmq_consumer PRIVATE -O3 -ffast-math -march=native)
    target_compile_options(cpp_timeseries_db PRIVATE -O3 -ffast-math -march=native)
endif()
