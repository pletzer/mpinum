import pnumpy
from pnumpy import CubeDecomp
import sys
import numpy
from mpi4py import MPI

"""
Apply the finite difference Laplacian operator in 2d
"""

# local rank and number of procs
rk = MPI.COMM_WORLD.Get_rank()
sz = MPI.COMM_WORLD.Get_size()

# global domain sizes
nx, ny = 12, 36

# domain sizes
xMin, xMax = 0.0, 1.0
yMin, yMax = 0.0, 1.0

dx, dy = (xMax - xMin)/float(nx), (yMax - yMin)/float(ny)

# axes, cell centered
xs = numpy.array([xMin + dx*(i+0.5) for i in range(nx)])
ys = numpy.array([yMin + dy*(j+0.5) for j in range(ny)])

# domain decomposition
dc = CubeDecomp(sz, (nx, ny))

# the decomp must be regular
if not dc.getDecomp():
    if rk == 0: 
        print('no decomp could be found, adjust the number of procs')
        MPI.Finalize()
        sys.exit(1)

# number of procs in each direction
npx, npy = dc.getDecomp()

# list of slices
slab = dc.getSlab(rk)

# starting/ending indices for local domain
iBeg, iEnd = slab[0].start, slab[0].stop
jBeg, jEnd = slab[1].start, slab[1].stop

# local variables
xx = numpy.outer( xs[iBeg:iEnd], numpy.ones( (ny/npy,), numpy.float64 ) )
yy = numpy.outer( numpy.ones( (nx/npx,), numpy.float64 ), ys[jBeg:jEnd] )

# local field
zz = numpy.sin(numpy.pi*xx) * numpy.cos(2*numpy.pi*yy)

# create and set distributed array
zda = pnumpy.gdaZeros( zz.shape, zz.dtype, numGhosts=1 )
zda[:] = zz

# compute the star Laplacian in the interior, this does not require
# any communication

laplaceZ = 4 * zda[:]

# local neighbour contributions, no communication
laplaceZ[1:  , :] -= zda[0:-1,:]
laplaceZ[0:-1, :] -= zda[1:  ,:]
laplaceZ[:, 1:  ] -= zda[:,0:-1]
laplaceZ[:, 0:-1] -= zda[:,1:  ]


# now compute and fill in the halo

# find the procs to the north, east, south, and west. This call will
# return None if there is no neighbour. 
noProc = dc.getNeighborProc(rk, ( 1,  0), periodic = (False, True)) 
soProc = dc.getNeighborProc(rk, (-1,  0), periodic = (False, True)) 
eaProc = dc.getNeighborProc(rk, ( 0,  1), periodic = (False, True)) 
weProc = dc.getNeighborProc(rk, ( 0, -1), periodic = (False, True))

# correct at the non-periodic boundaries
if noProc is None: 
    laplaceZ[-1,:] -= zda[-1,:]
if soProc is None:
    laplaceZ[0,:] -= zda[0,:]

# fetch the remote data in the halo of the neighbouring processor. When
# the first argument is None, this amounts to a no-op (zero data are 
# returned. Note that winID refers to the neighbour domain. For instance,
# the data to the west of the local domain correspond to the east halo
# on the neighbouring processor.
print('noProc = {0}, soProc = {1}, eaProc = {2}, weProc = = {3}'.format( \
    noProc, soProc, eaProc, weProc))
weZData = zda.getData(weProc, winID=(0, +1))
eaZData = zda.getData(eaProc, winID=(0, -1))
soZData = zda.getData(soProc, winID=(+1, 0))
noZData = zda.getData(noProc, winID=(-1, 0))

# finish the operator
weSlc = zda.getEllipsis(winID=(0, -1))
eaSlc = zda.getEllipsis(winID=(0, +1))
soSlc = zda.getEllipsis(winID=(-1, 0))
noSlc = zda.getEllipsis(winID=(+1, 0))

if weProc is not None:
    laplaceZ[weSlc] -= weZData
if eaProc is not None:
    laplaceZ[eaSlc] -= eaZData
if soProc is not None:
    laplaceZ[soSlc] -= soZData
if noProc is not None:
    laplaceZ[noSlc] -= noZData

# if you don't want to get a warning message
zda.free()

