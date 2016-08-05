# pnumpy
Parallel computing in N dimensions and in python

## Overview

pnumpy is a very lightweight implementation of distributed arrays,
which runs on architectures ranging from multi-core laptops to large
MPI clusters.  pnumpy is based on numpy and mpi4py and supports arrays in
any number of dimensions. Processes can access remote data using a "get" 
method. This can be used to access neighbor ghost data but is more 
flexible as it allows to access data from any process--not necessarily
a neighboring one. pnumpy is designed to work seamlessly with numpy's 
slicing operators ufunc, etc., making it easy to transition your code
from using numpy arrays to pnumpy arrays.

## How to get pnumpy

```bash
git clone https://github.com/pletzer/pnumpy.git
```

## How to build pnumpy

pnumpy requires:

 * python 2.7 or 3.5
 * numpy, e.g. 1.10
 * mpi4py, e.g. 2.0.0 (requires MPI library to be installed)


```bash
python setup.py install
```

or, if you need root access,

```bash
sudo python setup.py install
```

## How to test pnumpy

Run any file under tests/, e.g.

```bash
cd tests
mpiexec -n 4 python testDistArray.py
```