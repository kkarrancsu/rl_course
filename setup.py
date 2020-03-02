#!/usr/bin/env python3

import setuptools
import os

__author__ = 'Kiran Karra, Chace Ashcraft'
__email__ = 'kiran.karra@gmail.com, cc.ash.math@gmail.com'
__version__ = '0.0.1'

on_rtd = os.environ.get('READTHEDOCS') == 'True'

setuptools.setup(
    name='sorty',
    version=__version__,

    description='RL Course Environments & Agent',
    long_description='RL Course Environments & Agent',

    url = 'https://github.com/kkarrancsu/rl_course',

    author=__author__,
    author_email=__email__,

    license='Apache License 2.0',

    python_requires='>=3',
    packages=['sorty'],

    install_requires=[
        'gym',
    ],

    zip_safe=False
)
