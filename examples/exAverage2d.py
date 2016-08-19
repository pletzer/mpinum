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

# global domain sizes
nxG, nyG = 4, 8

# domain decomposition
dc = CubeDecomp(sz, (nxG, nyG))

# starting/ending indices for local domain
slab = dc.getSlab(rk)
iBeg, iEnd = slab[0].start, slab[0].stop
jBeg, jEnd = slab[1].start, slab[1].stop

# local domain sizes
nx, ny = iEnd  - iBeg, jEnd - jBeg

# the decomp must be regular
if not dc.getDecomp():
    if rk == 0: 
        print('no decomp could be found, adjust the number of procs')
        MPI.Finalize()
        sys.exit(1)

# create and set the input distributed array
inputData = pnumpy.gdaZeros((nx, ny), numpy.float32, numGhosts=1)
if rk == 0:
    inputData[-1, -1] = 1.0

# store the number of times a cell has an invalid neighbor so 
# we can correct the weights
numInvalidNeighbors = numpy.zeros((nx, ny), numpy.int32)

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
inputAvg = inputData.reduce(lambda x,y:x+y, 0., 0)
if rk == 0:
    inputAvg /= float(sz * nx * ny)
    print('input average: {0:.5f}'.format(inputAvg))

# use a star stencil (9 weights in 2d)

outputData = numpy.zeros((nx, ny), numpy.float32)

# zero displacement stencil
outputData[:] = inputData

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
    outputData[dst] += inputData[src]

    # remote updates
    src = domain.extract(disp).getSlice()
    dst = domain.extract(nisp).getSlice()
    neighRk = dc.getNeighborProc(rk, disp, periodic=notPeriodic)
    outputData[dst] += inputData.getData(neighRk, nisp)

    # will need to fix the weights when there is no neighbor
    if neighRk is None:
       numInvalidNeighbors[dst] += 3

#
# south-west, north-west, south-east, north-east
#

for disp in (-1, -1), (-1, 1), (1, -1), (1, 1):

    # negative of displacement
    nisp = tuple([-d for d in disp])

    # displacement purely along axis 0
    d0 = (disp[0], 0)
    n0 = tuple([-d for d in d0])

    # displacement purely along axis 1
    d1 = (0, disp[1])
    n1 = tuple([-d for d in d1])

    # local updates
    src = domain.shift(disp).getSlice()
    dst = domain.shift(nisp).getSlice()
    #print('disp = {} d0 = {} d1 = {} src = {} dst = {}'.format(disp, d0, d1, src, dst))
    outputData[dst] += inputData[src]

    # remote updates
    neighRk0 = dc.getNeighborProc(rk, d0, periodic=notPeriodic)
    neighRk1 = dc.getNeighborProc(rk, d1, periodic=notPeriodic)
    neighRk = dc.getNeighborProc(rk, disp, periodic=notPeriodic)
    #print('[{}] disp = {} neighRk = {}'.format(rk, disp, neighRk))
    data0 = inputData.getData(neighRk0, n0)
    data1 = inputData.getData(neighRk1, n1)
    data = inputData.getData(neighRk, nisp)

    # first axis
    src = domain.shift(d0).extract(d1).getSlice()
    dst = domain.shift(n0).extract(n1).getSlice()
    outputData[dst] += data1[src]
    # no need to correct numInvalidNeighbors since already taken into account

    # second axis
    src = domain.shift(d1).extract(d0).getSlice()
    dst = domain.shift(n1).extract(n0).getSlice()
    outputData[dst] += data0[src]
    # no need to correct numInvalidNeighbors since already taken into account

    # diagonal (mixed)
    src = domain.extract(d0).extract(d1).getSlice()
    dst = domain.extract(n0).extract(n1).getSlice()
    outputData[dst] += data
    if neighRk is None and neighRk0 is None and neighRk1 is None:
        #print('[{}] incrementating dst={} src={}'.format(rk, dst, src))
        # the diagonal direction is counted twice
        numInvalidNeighbors[dst] -= 1

# divide by the numbe of valid neighbors
outputData /= (9. - numInvalidNeighbors)

outputAvg = numpy.sum(outputData)
outAvg = numpy.sum(MPI.COMM_WORLD.gather(outputAvg, root=0))
if rk == 0:
    outAvg /= float(sz * nx * ny)
    print('output average: {0:.5f}'.format(outAvg))

print('[{}] input: \n {}'.format(rk, inputData))
print('[{}] output: \n {}'.format(rk, outputData))
#print('[{}] numInvalidNeighbors: \n {}'.format(rk, numInvalidNeighbors))

# if you don't want to get a warning message
inputData.free()
