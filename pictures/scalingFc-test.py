#!/usr/bin/env python

nprocs = [1, 2, 4, 8]

# exAverage2d 512 * 1024
times = [14.1, 5.8, 3.4, 4.4]

speedup = [times[0]/t for t in times]

from matplotlib import pylab
pylab.plot(nprocs, speedup, 'ko', nprocs, speedup, 'b-', nprocs, nprocs, 'c--')
pylab.title('fc-test')
pylab.xlabel('number of MPI processes')
pylab.ylabel('speedup')
pylab.axes([0, 8, 0, 5])

pylab.show()