import pnumpy
from pnumpy import CubeDecomp
import sys
import numpy
from mpi4py import MPI

"""
Apply the finite difference Laplacian operator in 3d
"""

# local rank and number of procs
rk = MPI.COMM_WORLD.Get_rank()
sz = MPI.COMM_WORLD.Get_size()

# domain sizes
n = 128
if len(sys.argv) > 1: 
    n = int(sys.argv[1])

# numbe rof times the Laplacian is applied
nTimes = 1
if len(sys.argv) > 2: 
    nTimes = int(sys.argv[2])    

nx, ny, nz = n, n, n
if rk == 0:
    print('Number of cells nx, ny, nz = {0}, {1}, {2}'.format(nx, ny, nz))
    print('Number of times Laplacian operator is applied = {0}'.format(nTimes))

# domain sizes
xMin, xMax = 0.0, 1.0
yMin, yMax = 0.0, 1.0
zMin, zMax = 0.0, 1.0

dx = (xMax - xMin)/float(nx)
dy = (yMax - yMin)/float(ny)
dz = (zMax - zMin)/float(nz)

# domain dc.sition
dc = CubeDecomp(sz, (nx, ny, nz))

# the dc.must be regular
if not dc.getDecomp():
    if rk == 0: 
        print('no decomp could be found, adjust the number of procs')
        MPI.Finalize()
        sys.exit(1)

# number of procs in each direction
npx, npy, npz = dc.getDecomp()
if rk == 0:
    print('Number of procs in x, y, z = {0}, {1}, {2}'.format(npx, npy, npz))

# list of slices
slab = dc.getSlab(rk)

# starting/ending indices for local domain
iBeg, iEnd = slab[0].start, slab[0].stop
jBeg, jEnd = slab[1].start, slab[1].stop
kBeg, kEnd = slab[2].start, slab[2].stop

# set the input field
f = numpy.zeros( (nx, ny, nz), numpy.float64 )

for i in range(iEnd - iBeg):
    x = xMin + dx*(iBeg + i + 0.5)
    for j in range(jEnd - jBeg):
        y = yMin + dy*(jBeg + j + 0.5)
        for k in range(kEnd - kBeg):
            z = zMin + dz*(kBeg + k + 0.5)
            f[i, j, z] = numpy.sin(numpy.pi*x) * numpy.cos(2*numpy.pi*y)

# compute the Laplacian
lapl = pnumpy.Laplacian(dc, periodic=(True, True, True))
time = MPI.Wtime()
for i in range(nTimes):
    if rk == 0:
        print('.', end='')
    fout = lapl.apply(f)
time = MPI.Wtime() - time
if rk == 0:
    print('')

times = MPI.COMM_WORLD.gather(time, 0)
if rk == 0:
    minTime = min(times)
    avgTime = numpy.sum(times)/float(sz)
    maxTime = max(times)
    print('Min/avg/max times: {0:.2f}/{1:.2f}/{2:.2f} s'.format(
           minTime, avgTime, maxTime))

# check
localChkSum = fout.sum()
chksum = numpy.sum(MPI.COMM_WORLD.gather(localChkSum, 0))
if rk == 0: 
    print('check sum = {}'.format(chksum))


