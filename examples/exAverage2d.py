import pnumpy
from pnumpy import CubeDecomp
from pnumpy import Partition
import sys
import numpy
from mpi4py import MPI

"""
Apply averaging sencil in 2d
"""

# local rank and number of procs
rk = MPI.COMM_WORLD.Get_rank()
sz = MPI.COMM_WORLD.Get_size()

# local domain sizes
nx, ny = 3, 3

# domain decomposition
dc = CubeDecomp(sz, (nx, ny))

# the decomp must be regular
if not dc.getDecomp():
    if rk == 0: 
        print('no decomp could be found, adjust the number of procs')
        MPI.Finalize()
        sys.exit(1)

# create and set the input distributed array
inputData = pnumpy.gdaZeros((nx, ny), numpy.float32, numGhosts=1 )
if rk == 0:
    inputData[0, 0] = 1.0

domain = Partition(2)

# the ghosted array only exposes west, east, south and north
# windows. Need to also export the corners
for disp in (-1, -1), (-1, 1), (1, -1), (1, 1):
    d0 = (disp[0], 0)
    d1 = (0, disp[1])
    n0 = (-disp[0], 0)
    n1 = (0, -disp[1])
    crner = domain.extract(n0).extract(n1).getSlice()
    inputData.expose(crner, disp)


# input average
inputAvg = inputData.reduce(lambda x,y:x+y)/float(sz * nx * ny)
if rk == 0:
    print('input average: {0:.5f}'.format(inputAvg))

# use a star stencil (9 weights in 2d)

outputData = numpy.zeros((nx, ny), numpy.float32)

# zero displacement stencil
outputData[:] = inputData / 9.

notPeriodic = (False, False)

#
# west, east, south and north
#

for disp in (-1, 0), (1, 0), (0, -1), (0, 1):

    # negative of disp
    nisp = tuple([-d for d in disp])

    # local updates
    src = domain.shift(disp).getSlice()
    dst = domain.shift(nisp).getSlice()
    print('disp = {} src = {} dst = {}'.format(disp, src, dst))
    outputData[dst] += inputData[src] / 9.

    # remote updates
    src = domain.extract(disp).getSlice()
    dst = domain.extract(nisp).getSlice()
    neighRk = dc.getNeighborProc(rk, disp, periodic=notPeriodic)
    outputData[dst] += inputData.getData(neighRk, nisp) / 9.

#
# south-west, north-west, south-east, north-east
#

for disp in []: #(-1, -1), (-1, 1), (1, -1), (1, 1):

    # negative of displacement
    nisp = tuple([-d for d in disp])

    # displacement purely along axis 0
    d0 = [0, 0]
    d0[0] = disp[0]
    d0 = tuple(d0)
    n0 = tuple([-d for d in d0])

    # displacement purely along axis 1
    d1 = [0, 0]
    d1[1] = disp[1]
    d1 = tuple(d1)
    n1 = tuple([-d for d in d1])

    # local updates
    src = domain.shift(disp).getSlice()
    dst = domain.shift(nisp).getSlice()
    outputData[dst] += inputData[src] / 9.

    # remote updates
    neighRk0 = dc.getNeighborProc(rk, d0, periodic=notPeriodic)
    neighRk1 = dc.getNeighborProc(rk, d1, periodic=notPeriodic)
    neighRk = dc.getNeighborProc(rk, disp, periodic=notPeriodic)
    data0 = inputData.getData(neighRk0, n0)
    data1 = inputData.getData(neighRk1, n1)
    data = inputData.getData(neighRk, nisp)

    src = domain.shift(d0).extract(d1).getSlice()
    dst = domain.shift(n0).extract(n1).getSlice()
    outputData[dst] += data1[src] / 9.

    src = domain.shift(d1).extract(d0).getSlice()
    dst = domain.shift(n1).extract(n0).getSlice()
    outputData[dst] += data0[src] / 9.

    src = domain.extract(d0).extract(d1).getSlice()
    dst = domain.extract(n0).extract(n0).getSlice()
    outputData[dst] += data[src] /9.


outputAvg = numpy.sum(outputData)
outAvgs = MPI.COMM_WORLD.gather(outputAvg, root=0)
if rk == 0:
    print('output average: {0:.5f}'.format(numpy.sum(outAvgs)))

print('[{}] input: \n {}'.format(rk, inputData))
print('[{}] output: \n {}'.format(rk, outputData))

# if you don't want to get a warning message
inputData.free()



