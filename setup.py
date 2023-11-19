#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup


setup(
    name='radiant',
    version='0.0.0',
    packages=[
        'radiate',
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'scipy',
        'matplotlib',
    ],
    maintainer='J. Keane Quigley',
    maintainer_email='james.quigley@st-hildas.ox.ac.uk',
    description='Multilevel Radial Basis Function Approximation of PDEs.',
    license='MIT',
    python_requires=">=3.10",
)
