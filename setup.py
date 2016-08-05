#!/usr/bin/env python

from setuptools import setup
import sys

# tests
try:
    import numpy
except:
    print("""
You must have numpy installed. 
You can download numpy from http://numpy.scipy.org/
""")
    sys.exit(1)

try:
    import mpi4py
except:
    print("""
You must have mpi4py installed. 
You can download mpi4py from http://code.google.com/p/mpi4py/
""")
    sys.exit(2)

setup(name = 'pnumpy',
      version = '1.2.0',
      description = 'A very lightweight implementation of distributed arrays',
      author = 'Alexander Pletzer',
      author_email = 'alexander@gokliya.net',
      url = 'https://github.com/pletzer/pnumpy.git',
      install_requires=['mpi4py>=2.0.0'],
      dependency_links=['https://github.com/mpi4py/mpi4py'],
      package_dir={'pnumpy': 'src'}, # the present directory maps to src 
      packages = ['pnumpy'],
)
