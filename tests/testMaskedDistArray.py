import pnumpy
import numpy
import unittest
from mpi4py import MPI
from functools import reduce

class TestMaskedDistArray(unittest.TestCase):
    """
    Test masked version of dist array
    """

    def setUp(self):
        pass

    def test0(self):
        """
        Test constructors
        """
        mask = numpy.zeros( (2, 3), numpy.bool_ )
        mask[0, 0] = 0
        da = pnumpy.mdaZeros( (2,3), numpy.float64, mask=mask )
        da = pnumpy.mdaOnes( (2,3), numpy.float64, mask=mask )
        da = pnumpy.mdaArray( [1,2,3], mask=numpy.array([0, 1, 1], numpy.bool_) )

    def test1d_1(self):
        """
        1D, float64
        """

        dtyp = numpy.float64

        # MPI stuff
        comm = MPI.COMM_WORLD
        rk = comm.Get_rank()
        sz = comm.Get_size()

        # create the dist array
        n = 10
        mask = numpy.zeros( (n,), numpy.bool_ )
        mask[-1] = 1 # last element is invalid
        da = pnumpy.mdaZeros( (n,), dtyp, mask=mask )
        # expose the last element
        da.expose( slce=(slice(-1, None, None),), winID='left' )
        # set data
        da[:] = 100*rk + numpy.array([i for i in range(n)], dtyp)
        # access remote data
        leftRk = (rk - 1) % sz
        print('proc {0} tries to access data and mask from {1}'.format(rk, leftRk))
        leftData = da.getData(pe=leftRk, winID='left')
        leftMask = da.getMask(pe=leftRk, winID='left')
        print('leftData for rank {0} = {1}'.format(rk, str(leftData)))
        print('leftMask for rank {0} = {1}'.format(rk, str(leftMask)))
        # check
        if leftRk < rk:
            self.assertEqual(leftData[0], da[-1] - 100)
        else:
            self.assertEqual(leftData[0], da[-1] + 100*(sz-1))
        # free
        da.free()

    def test1d_2(self):
        """
        1D, float32
        """

        dtyp = numpy.float32

        # MPI stuff
        comm = MPI.COMM_WORLD
        rk = comm.Get_rank()
        sz = comm.Get_size()

        # create the dist array
        n = 10
        mask = numpy.zeros( (n,), numpy.bool_ )
        da = pnumpy.mdaZeros( (n,), dtyp, mask=mask )
        # expose the last element
        da.expose( slce=(slice(-1, None, None),), winID='left' )
        # set data
        da[:] = 100*rk + numpy.array([i for i in range(n)], dtyp)
        # access remote data
        leftRk = (rk - 1) % sz
        print('proc %d tries to access data and mask from %d' % (rk, leftRk))
        leftData = da.getData(pe=leftRk, winID='left')
        leftMask = da.getMask(pe=leftRk, winID='left')
        print('leftData for rank {0} = {1}'.format(rk, str(leftData)))
        print('leftMask for rank {0} = {1}'.format(rk, str(leftMask)))
        # check
        if leftRk < rk:
            self.assertEqual(leftData[0], da[-1] - 100)
        else:
            self.assertEqual(leftData[0], da[-1] + 100*(sz-1))
        # free
        da.free()

    def test1d_3(self):
        """
        1D, int
        """

        dtyp = numpy.int64

        # MPI stuff
        comm = MPI.COMM_WORLD
        rk = comm.Get_rank()
        sz = comm.Get_size()

        # create the dist array
        n = 10
        da = pnumpy.mdaZeros( (n,), dtyp ) # all data are valid
        # expose the last element
        da.expose( slce=(slice(-1, None, None),), winID='left' )
        # set data
        da[:] = 100*rk + numpy.array([i for i in range(n)], dtyp)
        # access remote data
        leftRk = (rk - 1) % sz
        print('proc %d tries to access data and mask from %d' % (rk, leftRk))
        leftData = da.getData(pe=leftRk, winID='left')
        leftMask = da.getMask(pe=leftRk, winID='left')
        print('leftData for rank %d = %s' % (rk, str(leftData)))
        print('leftMask for rank %d = %s' % (rk, str(leftMask)))
        # check
        if leftRk < rk:
            self.assertEqual(leftData[0], da[-1] - 100)
        else:
            self.assertEqual(leftData[0], da[-1] + 100*(sz-1))
        # free
        da.free()

    def test2d_1(self):
        

        # create the dist array, the sizes are local to each processor
        mask = numpy.zeros( (2,3), numpy.bool_ )
        mask[1, 2] = 1
        da = pnumpy.mdaZeros( (2,3), numpy.float32, mask=mask )

        # processor rank and number of processes
        rk = da.rk
        nprocs = da.sz

        # expose sub-domains
        northSlab = ( slice(-1, None, None), slice(0, None, None) )
        da.expose( slce=northSlab, winID='n' )

        # set the data 
        da[:] = rk

        # access the remote data, collective operation
        northData = da.getData( (rk-1) % nprocs, winID='n' )
        northMask = da.getMask( (rk-1) % nprocs, winID='n' )

        # check sum
        checkSum = da.reduce(lambda x,y:x+y, rootPe=0)
        if checkSum is not None:
            exactCheckSum = reduce(lambda x,y:x+y,
                                   [rank for rank in range(nprocs)])
            ntot = reduce(lambda x,y:x*y, da.shape)
            exactCheckSum *= ntot
            self.assertEqual(checkSum, exactCheckSum) 

        # check 
        self.assertEqual(northData.min(), (rk - 1) % nprocs)
        self.assertEqual(northData.max(), (rk - 1) % nprocs)

        # clean up
        da.free()
       
if __name__ == '__main__':
    print("") # Spacer 
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMaskedDistArray)
    unittest.TextTestRunner(verbosity = 1).run(suite)
    MPI.Finalize()
    
