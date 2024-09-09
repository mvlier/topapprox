# run
# python setup.py build_ext --inplace

from setuptools import setup, Extension, find_packages
import pybind11
from pybind11.setup_helpers import Pybind11Extension


ext_modules = [
    Pybind11Extension(
        'topapprox.link_reduce',
        ['topapprox/link_reduce.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++11'],
    ),
]

setup(
    name='topapprox',
    author='Matias de Jong van Lier, Junyan Chu, Sebastían Elías Graiff Zurita, Shizuo Kaji',
    description='A module for topological low persistence filter',
    packages=find_packages(),
    ext_modules=ext_modules,
)
