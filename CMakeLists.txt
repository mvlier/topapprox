cmake_minimum_required(VERSION 3.18)
project(topapprox_native LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(link_reduce_cpp topapprox/link_reduce_cpp.cpp)
target_compile_features(link_reduce_cpp PRIVATE cxx_std_17)

if(MSVC)
  target_compile_options(link_reduce_cpp PRIVATE /O2 /EHsc /permissive-)
else()
  target_compile_options(link_reduce_cpp PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if(ipo_supported)
  set_property(TARGET link_reduce_cpp PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

install(TARGETS link_reduce_cpp LIBRARY DESTINATION topapprox RUNTIME DESTINATION topapprox)
