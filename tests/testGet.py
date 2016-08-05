#!/usr/bin/env python

import numpy
from mpi4py import MPI
import unittest

"""
Test one sided communication
"""

class TestGet(unittest.TestCase):

  def setUp(self):
    pass

  def test1(self):

    # MPI stuff
    comm = MPI.COMM_WORLD
    rk = comm.Get_rank()
    sz = comm.Get_size()
    procName = MPI.Get_processor_name()
    print('[{0}] {1}'.format(rk, procName))

    n = 10

    # create array
    srcBuffer = numpy.array([rk + 0.0] * n)
    dstBuffer = -1 * numpy.ones( srcBuffer.shape, srcBuffer.dtype )

    # create windows
    win = MPI.Win.Create(srcBuffer, comm=comm)

    # transfer data
    otherRk = (rk + 1) % sz
    win.Fence(MPI.MODE_NOPUT | MPI.MODE_NOPRECEDE)
    win.Get( [dstBuffer, MPI.DOUBLE], otherRk )
    win.Fence(MPI.MODE_NOSUCCEED)
    print('[{0}] data transfer from {1} succeeded'.format(rk, otherRk))

    # clean up 
    win.Free()

    # check 
    assert(numpy.sum(dstBuffer) == n*otherRk)

if __name__ == '__main__':
    print("") # Spacer 
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGet)
    unittest.TextTestRunner(verbosity = 1).run(suite)
    MPI.Finalize()
