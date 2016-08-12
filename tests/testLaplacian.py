#!/usr/bin/env python

"""
Testing Laplacian discretization in 1d
@author Alexander Pletzer
"""

import sys
import unittest
from mpi4py import MPI
import numpy
from pnumpy import CubeDecomp
from pnumpy import Laplacian

class TestLaplacian(unittest.TestCase):

  def setUp(self):
    # number of procs
    self.sz = MPI.COMM_WORLD.Get_size()
    # MPI rank
    self.rk = MPI.COMM_WORLD.Get_rank()

  def test1d(self):

    n = 8

    # global number of cells
    ns = (n,)

    # domain decomposition
    dc = CubeDecomp(self.sz, ns)
    if not dc.getDecomp():
        print('*** ERROR Invalid domain decomposition -- rerun with different sizes/number of procs')
        sys.exit(1)
    ndims = dc.getNumDims()

    # local start/end grid indices
    slab = dc.getSlab(self.rk)

    # global domain boundaries
    xmins = numpy.array([0.0 for i in range(ndims)])
    xmaxs = numpy.array([1.0 for i in range(ndims)])

    # local cell centered coordinates
    axes = []
    hs = []
    for i in range(ndims):
        ibeg, iend = slab[i].start, slab[i].stop
        h = (xmaxs[i] - xmins[i]) / float(ns[i])
        ax = xmins[i] + h*(numpy.arange(ibeg, iend) + 0.5)
        hs.append(h)
        axes.append(ax)

    lapl = Laplacian(dc, periodic=(False,))

    # set the input function
    inp = 0.5 * axes[0]**2
    #print('[{0}] inp = {1}'.format(self.rk, str(inp)))

    out = lapl.apply(inp) / hs[0]**2
    #print('[{0}] out = {1}'.format(self.rk, str(out)))

    # check sum
    localChkSum = numpy.sum(out.flat)
    chksum = numpy.sum(MPI.COMM_WORLD.gather(localChkSum, 0))
    if self.rk == 0: 
        print('test1d check sum = {}'.format(chksum))
        self.assertLessEqual(abs(chksum - -28.25), 1.e-10)


  def test2d(self):

    n = 8

    # global number of cells
    ns = (n, n)

    # domain decomposition
    dc = CubeDecomp(self.sz, ns)
    if not dc.getDecomp():
        print('*** ERROR Invalid domain decomposition -- rerun with different sizes/number of procs')
        sys.exit(1)
    ndims = dc.getNumDims()

    # local start/end grid indices
    slab = dc.getSlab(self.rk)

    # global domain boundaries
    xmins = numpy.array([0.0 for i in range(ndims)])
    xmaxs = numpy.array([1.0 for i in range(ndims)])

    # local cell centered coordinates
    axes = []
    hs = []
    nsLocal = []
    for i in range(ndims):
        ibeg, iend = slab[i].start, slab[i].stop
        nsLocal.append(iend - ibeg)
        h = (xmaxs[i] - xmins[i]) / float(ns[i])
        ax = xmins[i] + h*(numpy.arange(ibeg, iend) + 0.5)
        hs.append(h)
        axes.append(ax)

    lapl = Laplacian(dc, periodic=(False, False))

    # set the input function
    xx = numpy.outer(axes[0], numpy.ones((nsLocal[1],)))
    yy = numpy.outer(numpy.ones((nsLocal[0],)), axes[1])
    inp = 0.5 * xx * yy ** 2
    #print('[{0}] inp = {1}'.format(self.rk, str(inp)))

    out = lapl.apply(inp) / hs[0]**2
    #print('[{0}] out = {1}'.format(self.rk, str(out)))

    # check sum
    localChkSum = numpy.sum(out.flat)
    chksum = numpy.sum(MPI.COMM_WORLD.gather(localChkSum, 0))
    if self.rk == 0: 
        print('test2d check sum = {}'.format(chksum))
        self.assertLessEqual(abs(chksum - -198.0), 1.e-10)


if __name__ == '__main__': 
  print("") # Spacer 
  suite = unittest.TestLoader().loadTestsFromTestCase(TestLaplacian)
  unittest.TextTestRunner(verbosity = 1).run(suite)
  MPI.Finalize()
