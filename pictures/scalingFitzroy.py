#!/usr/bin/env python
import matplotlib
matplotlib.rcParams.update({'font.size': 22})

nprocs = [1, 4, 8, 16, 32]

#  exLaplace3d 1024 10
times = [930, 343, 307, 161, 45]

speedup = [times[0]/t for t in times]

from matplotlib import pylab
pylab.plot(nprocs, speedup, 'ko', nprocs, speedup, 'b-', [1, 20], [1, 20], 'c--')
pylab.title('FitzRoy exLaplace3d 1024 10')
pylab.xlabel('number of MPI processes')
pylab.ylabel('speedup')
pylab.axes([1, 32, 1, 20])

pylab.show()
