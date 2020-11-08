#!/usr/bin/env python
import matplotlib
matplotlib.rcParams.update({'font.size': 22})

nprocs = [1, 2, 4, 8, 16, ]

#  exLaplace3d 512 10
times = [57.89, 30.1, 15.707, 11.4, 5.92, ]

speedup = [times[0]/t for t in times]

from matplotlib import pylab
pylab.plot(nprocs, speedup, 'ko', nprocs, speedup, 'b-', [1, 10], [1, 10], 'c--')
pylab.title('Mahuika exLaplace3d 512 10')
pylab.xlabel('number of MPI processes')
pylab.ylabel('speedup')
pylab.axes([1, 16, 1, 16])

pylab.show()
