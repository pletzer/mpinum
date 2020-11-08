#!/usr/bin/env python
import matplotlib
matplotlib.rcParams.update({'font.size': 22})

nprocs = [1, 2, 4, 8, 20, ]

#  exLaplace3d 512 10
times = [12*60+48.6, 9*60+51.1, 5*60+18.33, 2*60+49.0, 1*60+19.7, ]

speedup = [times[0]/t for t in times]

from matplotlib import pylab
pylab.plot(nprocs, speedup, 'ko', nprocs, speedup, 'b-', [1, 10], [1, 10], 'c--')
pylab.title('Maui ancil exLaplace3d 1000 20')
pylab.xlabel('number of MPI processes')
pylab.ylabel('speedup')
pylab.axes([1, 20, 1, 20])

pylab.show()
