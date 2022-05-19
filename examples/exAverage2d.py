import mpinum
from mpinum import CubeDecomp
from mpinum import Partition
import sys
import numpy
from mpi4py import MPI

"""
Apply 9 point averaging sencil in 2d
"""

def plot(nxG, nyG, iBeg, iEnd, jBeg, jEnd, data, title=''):
    """
    Plot distributed array
    @param nxG number of global cells in x
    @param nyG number of global cells in y
    @param iBeg global starting index in x
    @param iEnd global ending index in x
    @param jBeg global starting index in y
    @param jEnd global ending index in y
    @param data local array
    @param title plot title
    """
    sz = MPI.COMM_WORLD.Get_size()
    rk = MPI.COMM_WORLD.Get_rank()
    iBegs = MPI.COMM_WORLD.gather(iBeg, root=0)
    iEnds = MPI.COMM_WORLD.gather(iEnd, root=0)
    jBegs = MPI.COMM_WORLD.gather(jBeg, root=0)
    jEnds = MPI.COMM_WORLD.gather(jEnd, root=0)
    arrays = MPI.COMM_WORLD.gather(numpy.array(data), root=0)
    if rk == 0:
        bigArray = numpy.zeros((nxG, nyG), data.dtype)
        for pe in range(sz):
            bigArray[iBegs[pe]:iEnds[pe], jBegs[pe]:jEnds[pe]] = arrays[pe]
        from matplotlib import pylab
        pylab.pcolor(bigArray.transpose())
        # add the decomp domains
        for pe in range(sz):
            pylab.plot([iBegs[pe], iBegs[pe]], [0, nyG - 1], 'w--')
            pylab.plot([0, nxG - 1], [jBegs[pe], jBegs[pe]], 'w--')
        pylab.title(title)
        pylab.show()

def setValues(nxG, nyG, iBeg, iEnd, jBeg, jEnd, data):
    """
    Set setValues
    @param nxG number of global cells in x
    @param nyG number of global cells in y
    @param iBeg global starting index in x
    @param iEnd global ending index in x
    @param jBeg global starting index in y
    @param jEnd global ending index in y
    @param data local array
    """
    nxGHalf = nxG/2.
    nyGHalf = nyG/2.
    nxGQuart = nxGHalf/2.
    nyGQuart = nyGHalf/2.
    for i in range(data.shape[0]):
        iG = iBeg + i
        di = iG - nxG
        for j in range(data.shape[1]):
            jG = jBeg + j
            dj = jG - 0.8*nyG
            data[i, j] = numpy.floor(1.9*numpy.exp(-di**2/nxGHalf**2 - dj**2/nyGHalf**2))

# local rank and number of procs
rk = MPI.COMM_WORLD.Get_rank()
sz = MPI.COMM_WORLD.Get_size()

# global domain sizes
nxG, nyG = 128, 256

# domain decomposition
dc = CubeDecomp(sz, (nxG, nyG))

# starting/ending global indices for local domain
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
inputData = mpinum.gdaZeros((nx, ny), numpy.float32, numGhosts=1)
setValues(nxG, nyG, iBeg, iEnd, jBeg, jEnd, inputData)

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

plot(nxG, nyG, iBeg, iEnd, jBeg, jEnd, inputData, 'input')
plot(nxG, nyG, iBeg, iEnd, jBeg, jEnd, outputData, 'output')

#print('[{}] input: \n {}'.format(rk, inputData))
##print('[{}] numInvalidNeighbors: \n {}'.format(rk, numInvalidNeighbors))

# if you don't want to get a warning message
inputData.free()
