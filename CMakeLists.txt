cmake_minimum_required(VERSION 3.26...3.30)
project(topapprox_native LANGUAGES CXX)

if(NOT SKBUILD)
  message(WARNING "This CMake project is meant to be driven by scikit-build-core; "
                  "configuring it directly will most likely fail.")
endif()

# nanobind's stable ABI (abi3) support relies on FindPython's
# Development.SABIModule component, which requires CMake >= 3.26. The component
# is optional so that builds on CPython < 3.12 (where there is no stable ABI)
# still succeed and simply produce a version-specific extension.
find_package(Python 3.10 REQUIRED
  COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Locate the nanobind CMake package shipped inside the installed nanobind wheel.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# STABLE_ABI: build a single abi3 extension on CPython >= 3.12 (no-op below).
# NB_STATIC: statically link nanobind's runtime into this single extension.
# NOMINSIZE: keep release -O2/-O3 instead of nanobind's default -Os, since these
#            are performance-sensitive numeric kernels.
# NB_SUPPRESS_WARNINGS: treat nanobind/Python headers as system includes.
nanobind_add_module(link_reduce_cpp
  STABLE_ABI
  NB_STATIC
  NOMINSIZE
  NB_SUPPRESS_WARNINGS
  topapprox/link_reduce_cpp.cpp)

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
