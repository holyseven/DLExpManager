#!/usr/bin/env python3

import os
from setuptools import setup, find_packages
import sideman

with open("requirements.txt") as fin:
    REQUIRED_PACKAGES = fin.read()

# get key package details from py_pkg/__version__.py
about = {
    '__title__': 'sideman',
    '__description__': 'Simple Deep Learning Experiment Manager',
    '__version__': sideman.__version__
}  # type: ignore
here = os.path.abspath(os.path.dirname(__file__))
# with open(os.path.join(here, 'py_pkg', '__version__.py')) as f:
#     exec(f.read(), about)

# load the README file and use it as the long_description for PyPI
#with open('README.md', 'r') as f:
#    readme = f.read()

# package configuration - for reference see:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#id9
long_description = 'Home-page: https://github.com/holyseven/DLExpManager \
Author: Holyseven  \
Author-email: holyseven@hotmail.com \
License: Apache 2.0 \
Description: Deep Learning Experiment Manager.'

setup(
    name=about['__title__'],
    description=about['__description__'],
    long_description=long_description,
    long_description_content_type='text/plain',
    version=about['__version__'],
    #url=about['__url__'],
    packages=find_packages(),
    python_requires=">=3.7.*",
    install_requires=REQUIRED_PACKAGES,
    license='Apache 2.0',
    zip_safe=False
)
