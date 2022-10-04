from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy as np
import sys
import runpy
import pathlib

root = pathlib.Path(__file__).parent
version = runpy.run_path(str(root / "gail_tf2" / "version.py"))["version"]


setup(
    name='gail_tf2',
    version=version,
    description='Tensorflow 2 version of Generative Adversarial Imitation Learning',
    license='MIT',
    packages=find_packages()
)
