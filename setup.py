#!/usr/bin/env python

from setuptools import setup
import sys

setup(name = 'pnumpy',
      version = '1.4.0',
      description = 'A very lightweight implementation of distributed arrays',
      author = 'Alexander Pletzer',
      author_email = 'alexander@gokliya.net',
      url = 'https://github.com/pletzer/pnumpy.git',
      install_requires=['numpy>=1.8', 'mpi4py>=2.0.0'],
      dependency_links=['https://github.com/mpi4py/mpi4py'],
      package_dir={'pnumpy': 'src'}, # the present directory maps to src 
      packages = ['pnumpy'],
)
