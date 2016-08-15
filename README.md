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
 * MPI library (e.g. MPICH2 1.4.1p1 that comes with Anaconda)
 * mpi4py, e.g. 2.0.0


```bash
python setup.py install
```

or, if you need root access,

```bash
sudo python setup.py install
```

Alternatively you can use 
```python 
pip install pnumpy
```

or, if you're using Anaconda, 
```python
conda install pnumpy
```

## How to test pnumpy

Run any file under tests/, e.g.

```bash
cd tests
mpiexec -n 4 python testDistArray.py
```

## How to use pnumpy

### A lightweight extension to numpy arrays

Think of pnumpy arrays as standard numpy arrays with additional data members and methods to access neighboring data. 
To create a ghosted distributed array (gda) use:

```python
from pnumpy import gdaZeros
da = gdaZeros((4, 5), numpy.float32, numGhosts=1)
```

The above creates a 4 x 5 float32 array -- the syntax should be familiar to anyone using 
numpy arrays. 

Pnumpy distributed arrays are standard arrays except for additional methods and the fact 
that each MPI process holds its own data. As such, all numpy operations 
apply to pnumpy ghosted distributed arrays with no change and this includes slicing.

All slicing operations are with respect to the local array indices.

In the above, numGhosts describes the thickness of the halo region, i.e. the slice of 
data inside the array that can be accessed by other processes. A value of numGhosts = 1 means 
the halo has depth of one, standard finite differencing stencils require numGhosts = 1.

For a 2D array, the halo can be broken into four regions: 
da[:numGhosts, :], da[-numGhosts:, :], da[:, :numGhosts] and da[:, -numGhosts:].
(In n-dimensions there are 2n regions.) Pnumpy identifies each halo region
 with a tuple: (1, 0) for east, (-1, 0) for west, (0, 1) for north and (0, -1) for south. 

To access data on the south region of remote process otherRk, use
```python
southData = da.getData(otherRk, winID=(0, -1))
```

### Using a regular domain decomposition

The above will work for any domain decomposition, not necessarily a regular one. In the cases where a global array is split in 
uniform chunks of data, otherRk can be inferred from the local rank and an offset vector:

```python
from pnumpy import CubeDecomp
decomp = CubeDecomp(numProcs, dims)
...
otherRk = decomp.getNeighborProc(self, da.getMyMPIRank(), offset=(0, 1), periodic=(True, False))
```

where numProcs is the number of processes, dims are the global array dimensions and periodic is a tuple of 
True/False values indicating whether the domain is periodic or not. In the case where there is no neighbour rank (because the
local da.getMyMPIRank() rank lies at the boundary of the domain), then getNeighborProc may return None. In this case getData will also return None. 

### A very high level

For the Laplacian stencil, one may consider using 

```python
from pnumpy import Laplacian
lapl = Laplacian(decomp, periodic=(True, False))
```

Applying the Laplacian stencil to any numpy-like array inp then involves:
```python
out = lapl.apply(inp)
```
