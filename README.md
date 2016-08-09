# pnumpy
Parallel computing in N dimensions made easy in Python

## Overview

pnumpy is a very lightweight implementation of distributed arrays,
which runs on architectures ranging from multi-core laptops to large
MPI clusters.  pnumpy is based on numpy and mpi4py and supports arrays in
any number of dimensions. Processes can access remote data using a "getData" 
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

## How to use pnumpy

### A lightweight extension to numpy arrays

Think of numpy arrays with additional data members and methods to access neighboring data. To create a ghosted distributed array (gda):

```python
from pnumpy import gdaZeros

da = gdaZeros( (4, 5), numpy.float32, numGhosts=1 )
```

The above syntax should be familiar to anyone using numpy arrays. Each MPI process will get its own version of the array. Note that indexing is local to the array stored on a given process. This means that da[0, 0] will the first element on 
that process and da[-1, :] will the last row (following the usual numpy indexing rule).

It also means that pnumpy pnumpy does not assume a domain decomposition. The relationship between the arrays stored on different processes can be arbotrary, not necessarily a regular domain decomposition. 

The data stored on each process can be set using indexing, slicing and ellipses. For instance:

```python
rk = da.getMPIRank()
da[...] = 100 * rk
``` 

Option numGhosts describes the thickness of the halo region, i.e. the slice of data inside the array that can be accessed 
by neighboring processes. numGhosts = 1 means that the halo has depth of one. For a 2D array (as in this case), the halo 
can be broken into four slices: da[0, :], da[-1, :], da[:, 0] and da[:, -1] representing the west, east, south and north
sides of the array. Because the terminology of north, east, etc. does not extend to n-dimensional arrays, pnumpy denotes 
each side by a tuple (1, 0) for north, (0, 1) for east, (-1, 0) for south and (0, -1) for west. 

Any process can fetch the ghost data stored on another process (not necessarily the neighbor one) using:

```python
otherData = da.getData(otherMPIRank, side=(0, -1))
```

where otherMPIRank is the other process's MPI rank. Note that this is a collective operation -- all ranks are involved. 
Passing None in place of otherMPIRank is a no-operation -- this is useful if some processes don't have a neighbor.
