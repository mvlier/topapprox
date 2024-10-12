# run
# python setup.py build_ext --inplace

from setuptools import setup, Extension, find_packages
import pybind11
from pybind11.setup_helpers import Pybind11Extension

import os
import platform

compiler = os.environ.get('CXX', 'g++')
if platform.system() == 'Darwin':  # macOS
    extra_compile_args = ['-std=c++11', '-mmacosx-version-min=10.9']
else:
    extra_compile_args = ['-std=c++11']

ext_modules = [
    Pybind11Extension(
        'topapprox.link_reduce_cpp',
        ['topapprox/link_reduce_cpp.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args = extra_compile_args + ['-O3', '-Wall', '-shared'],
    ),
]

setup(
    name='topapprox',
    author='Matias de Jong van Lier, Junyan Chu, Sebastían Elías Graiff Zurita, Shizuo Kaji',
    description='A module for topological low persistence filter',
    packages=find_packages(),
    ext_modules=ext_modules,
    zip_safe=False,  # Ensure package is not installed as a zip
)
