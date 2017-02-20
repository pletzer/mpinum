#!/usr/bin/env python
import matplotlib
matplotlib.rcParams.update({'font.size': 22})

nprocs = [1, 2, 4, 8, 16, 32]

#  exLaplace3d 512 20
times = [2*60 + 21.4, 60 + 23.4, 48.4, 29.6, 19.7, 15.1]

speedup = [times[0]/t for t in times]

from matplotlib import pylab
pylab.plot(nprocs, speedup, 'ko', nprocs, speedup, 'b-', [1, 10], [1, 10], 'c--')
pylab.title('Pan Ivy Bridge exLaplace3d 512 20')
pylab.xlabel('number of MPI processes')
pylab.ylabel('speedup')
pylab.axes([1, 32, 1, 10])

pylab.show()
