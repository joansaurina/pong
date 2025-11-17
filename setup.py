#!/usr/bin/env python3
"""
Setup script for pong
"""
from setuptools import setup
from os import path

# Get the long description from the README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get version from VERSION file
with open(path.join(this_directory, 'VERSION')) as f:
    version = f.read().strip()

setup(
    name='pong',
    version=version,
    description='pong: fast visualization and analysis of population structure',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Aaron Behr, K. Liu, T. Devlin, G. Liu-Fang, and S. Ramachandran',
    url='https://github.com/ramachandran-lab/pong',
    python_requires='>=3.7',
    packages=['pong'],
    package_dir={'pong': '.'},
    package_data={
        'pong': [
            'VERSION',
            'templates/*.html',
            'static/**/*',
            'license/*.txt',
        ],
    },
    install_requires=[
        'numpy>=1.19',
        'munkres>=1.1',
        'networkx>=2.5',
    ],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'pong=pong.main:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
    ],
)

