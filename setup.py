from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy as np
import sys
import runpy
import pathlib

root = pathlib.Path(__file__).parent
version = 1.0

setup(
    name='GAIL_TF2'
    version=version,
    description='GAIL',
    author='Michael Drolet',
    author_email='mdrolet@asu.edu',
    license='MIT',
    packages=find_packages()
)
