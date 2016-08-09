#!/usr/bin/env python

"""
Upwind discretization of advection equation
@author Alexander Pletzer
"""

import pnumpy
import numpy
import sys
from mpi4py import MPI
import sys
import operator

import unittest

class Upwind: 

  def __init__(self, velocity, lengths, numCells):

    self.rk = MPI.COMM_WORLD.Get_rank()
    self.sz = MPI.COMM_WORLD.Get_size()

    # decomposition
    self.dc = pnumpy.CubeDecomp(self.sz, numCells)
    if not self.dc.getDecomp():
      print('*** No uniform decomposition could be found for {0} processes'.format(self.sz))
      print('*** Please ajust the number of cells {0}'.format(numCells))
      sys.exit(1)

    # begin/end indices of local sub-domain
    self.localSlices = self.dc.getSlab(self.rk)
    self.iBeg = numpy.array([s.start for s in self.localSlices])
    self.iEnd = numpy.array([s.stop for s in self.localSlices])
    self.nsLocal = numpy.array([s.stop - s.start for s in self.localSlices])
    print('[{0}] local number of cells: {1}'.format(self.rk, self.nsLocal))

    # global number of cells
    self.numCells = numCells

    self.ndims = 3
    self.deltas = numpy.zeros( (self.ndims,), numpy.float64 )
    self.upDirection = numpy.zeros( (self.ndims,), numpy.float64 )
    self.v = velocity
    self.lengths = lengths

    # number of local field values
    self.ntot = 1
    for j in range(self.ndims):
      self.upDirection[j] = -1
      if velocity[j] < 0.: self.upDirection[j] = +1
      self.deltas[j] = lengths[j] / numCells[j]
      self.ntot *= self.nsLocal[j]

    self.coeff = self.v * self.upDirection / self.deltas

    # initializing the field
    self.f = pnumpy.ghZeros( self.nsLocal, numpy.float64, numGhosts=1 )
    self.fOld = pnumpy.ghZeros( self.nsLocal, numpy.float64, numGhosts=1 )

    # initialize lower corner to one
    if self.rk == 0:
      self.f[0, 0, 0] = 1

    # get the neighboring ranks
    self.neighbSide = [[] for i in range(self.ndims)]
    direction = numpy.array([0] * self.ndims)
    self.neighbRk = numpy.array([0] * self.ndims)
    periodic = [True for i in range(self.ndims)]
    for i in range(self.ndims):
      direction[i] = self.upDirection[i]
      self.neighbRk[i] = self.dc.getNeighborProc(self.rk, direction, periodic=periodic)
      self.neighbSide[i] = tuple(-direction)
      direction[i] = 0

  def advect(self, deltaTime):
    """
    Advance the field by one time step
    """

    self.fOld[:] = self.f
    c = deltaTime * numpy.sum(self.coeff)

    # handle all local computations first
    self.f += c*self.fOld

    self.f[1:, :, :] -= deltaTime*self.coeff[0]*self.fOld[:-1, :, :]
    self.f[:, 1:, :] -= deltaTime*self.coeff[1]*self.fOld[:, :-1, :]
    self.f[:, :, 1:] -= deltaTime*self.coeff[2]*self.fOld[:, :, :-1]

    # fetch neighboring data. This is the only place where there is communication
    self.f[:1, :, :] -= deltaTime*self.coeff[0]* \
                        self.fOld.getData(self.neighbRk[0], self.neighbSide[0])
    self.f[:, :1, :] -= deltaTime*self.coeff[1]* \
                        self.fOld.getData(self.neighbRk[1], self.neighbSide[1])
    self.f[:, :, :1] -= deltaTime*self.coeff[2]* \
                        self.fOld.getData(self.neighbRk[2], self.neighbSide[2])

  def checksum(self):
    return self.f.reduce(operator.add, 0.0, rootPe=0)

  def printOut(self):
    for i in range(len(self.f)):
      print('{0} {1}'.format(i, self.f[i]))

  def __del__(self):
    self.f.free()
    self.fOld.free()

  def gatherRoot(self):
    """
    Gather the data on process root
    @return array on rank 0, None on other ranks
    """
    res = None
    if self.rk == 0:
      res = numpy.zeros(self.numCells, numpy.float64)

    fRoot = MPI.COMM_WORLD.gather(self.f, root=0)

    if self.rk == 0:
      for rk in range(self.sz):
        slab = self.dc.getSlab(rk)
        res[slab] = fRoot[rk]

    return res

############################################################################################################

class TestUpwind(unittest.TestCase):

  def setUp(self):
    pass

  def test64Cells10Steps(self):

    ndims = 3
    numCells = [64, 64, 64]
    numTimeSteps = 10

    velocity = numpy.array([1., 1., 1.])
    lengths = numpy.array([1., 1., 1.])

    # compute dt 
    courant = 0.1
    dt = float('inf')
    for j in range(ndims):
      dx = lengths[j]/ float(numCells[j])
      dt = min(courant * dx / velocity[j], dt)

    up = Upwind(velocity, lengths, numCells)
    if up.rk == 0: 
      print("number of cells: {0}".format(numCells))

    tic, tac = 0, 0
    if up.rk == 0: 
      tic = MPI.Wtime()

    # time iterations
    for i in range(numTimeSteps):
      up.advect(dt)

    if up.rk == 0:
      toc = MPI.Wtime()
      print('Wall clock time spent in advection loop: {0} [sec]'.format(toc - tic))

    chksum = up.checksum()
    if up.rk == 0:
      error = chksum - 1.0
      print('check sum error {0}'.format(error))
      self.assertLessEqual(abs(error), 1.e-8)

if __name__ == '__main__': 
  print("") # Spacer 
  suite = unittest.TestLoader().loadTestsFromTestCase(TestUpwind)
  unittest.TextTestRunner(verbosity = 1).run(suite)
  MPI.Finalize()
