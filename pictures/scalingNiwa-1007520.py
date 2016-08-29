#!/usr/bin/env python

nprocs = [1, 2, 4, ]# 8]

# exlaplacian3d.py 512 1 step float32
times = [5*60+3.7, 2*60+37.54, 1*60+18.598, ]# 1*60+34.35]

speedup = [times[0]/t for t in times]

from matplotlib import pylab
pylab.plot(nprocs, speedup, 'ko', nprocs, speedup, 'b-', nprocs, nprocs, 'c--')
pylab.title('exLaplacian3d: Intel(R) Xeon(R) CPU E5-1630 v3 @ 3.70GHz')
pylab.xlabel('number of MPI processes')
pylab.ylabel('speedup')
#pylab.axes([0, 4, 0, 5])

pylab.show()
